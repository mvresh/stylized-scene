import { useEffect, useMemo, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import {
  BufferGeometry,
  DoubleSide,
  Float32BufferAttribute,
  Object3D,
  Plane,
  Vector3,
} from "three";
import { MeshStandardNodeMaterial } from "three/webgpu";
import {
  attribute,
  cross,
  deltaTime,
  float,
  Fn,
  hash,
  If,
  instancedArray,
  instanceIndex,
  Loop,
  length,
  max,
  mix,
  normalize,
  positionLocal,
  sin,
  time,
  uniform,
  vec3,
} from "three/tsl";

// A starling murmuration simulated entirely on the GPU with TSL compute, sized
// to this scene's ~40-unit world (ground spans -20..20). Two storage buffers
// (position, velocity) are advanced each frame by a compute kernel running
// classic Reynolds boids (separation / alignment / cohesion) + a soft spherical
// boundary, a hard ground floor, a shared wind drift/swirl, and a mouse-driven
// predator. The birds are one instanced body+wings mesh whose vertex stage reads
// the buffers, orients each bird along its velocity and flaps its wings.

const COUNT = 1600;

// World tuning.
const CENTER: [number, number, number] = [0, 13, -2];
const SPREAD_XZ = 15;
const SPREAD_Y = 5;
const BOUND_R = 16; // soft spherical cage radius
const FLOOR_Y = 4.0; // birds never dip below this

const MAX_SPEED = 7.5;
const MIN_SPEED = 3.2;

// Looser flock: bigger personal space, weaker pull together.
const SEP_DIST = 1.1;
const ALI_DIST = 2.4;
const COH_DIST = 3.2;

const SEP_W = 5.0;
const ALI_W = 0.9;
const COH_W = 0.8;
const BOUND_W = 9.0;
const FLOOR_W = 40.0;

const WIND_GAIN = 6.0; // how strongly the shared wind nudges the flock
const SWIRL_W = 1.6; // gentle rotation around the flock center

const PRED_RADIUS = 6.0;
const PRED_STRENGTH = 90.0;

const BIRD_SCALE = 0.42;
const FLAP_SPEED = 20.0;
const FLAP_AMT = 0.9;

const FAR_AWAY = 1e4;

// Body + swept wings, pointing along +Z. aWing marks the flapping wing tips
// (1) vs the rigid body (0). A small vertical tail fin keeps it from reading as
// a flat sheet from above.
function makeBirdGeometry() {
  // prettier-ignore
  const data: Array<[number, number, number, number]> = [
    // body (nose, raised tail fin, tail)
    [0, 0, 0.55, 0], [0, 0.16, -0.4, 0], [0, 0, -0.55, 0],
    // left wing (root front, tip, root back)
    [0, 0, 0.05, 0], [-0.9, 0, -0.12, 1], [0, 0, -0.32, 0],
    // right wing (root back, tip, root front)
    [0, 0, -0.32, 0], [0.9, 0, -0.12, 1], [0, 0, 0.05, 0],
  ];
  const pos: number[] = [];
  const wing: number[] = [];
  for (const [x, y, z, w] of data) {
    pos.push(x, y, z);
    wing.push(w);
  }
  const g = new BufferGeometry();
  g.setAttribute("position", new Float32BufferAttribute(pos, 3));
  g.setAttribute("aWing", new Float32BufferAttribute(wing, 1));
  g.computeVertexNormals();
  g.scale(BIRD_SCALE, BIRD_SCALE, BIRD_SCALE);
  return g;
}

type Props = {
  windAngle: number;
  windStrength: number;
};

export function Flock({ windAngle, windStrength }: Props) {
  const gl = useThree((s) => s.gl);
  const camera = useThree((s) => s.camera);
  const pointer = useThree((s) => s.pointer);
  const raycaster = useThree((s) => s.raycaster);

  const geometry = useMemo(makeBirdGeometry, []);

  const sim = useMemo(() => {
    const positionBuffer = instancedArray(COUNT, "vec3");
    const velocityBuffer = instancedArray(COUNT, "vec3");

    const center = uniform(vec3(...CENTER));
    const windVec = uniform(vec3(0, 0, 0));
    const predator = uniform(vec3(FAR_AWAY, FAR_AWAY, FAR_AWAY));

    const initKernel = Fn(() => {
      const i = instanceIndex;
      const rx = hash(i.add(0.13)).sub(0.5);
      const ry = hash(i.add(1.71)).sub(0.5);
      const rz = hash(i.add(4.27)).sub(0.5);
      positionBuffer
        .element(i)
        .assign(
          vec3(rx.mul(SPREAD_XZ), ry.mul(SPREAD_Y), rz.mul(SPREAD_XZ)).add(center)
        );
      const vx = hash(i.add(7.3)).sub(0.5);
      const vy = hash(i.add(9.1)).sub(0.5);
      const vz = hash(i.add(11.9)).sub(0.5);
      velocityBuffer.element(i).assign(normalize(vec3(vx, vy, vz)).mul(MAX_SPEED));
    })().compute(COUNT);

    const updateKernel = Fn(() => {
      const i = instanceIndex;
      const pos = positionBuffer.element(i).toVar();
      const vel = velocityBuffer.element(i).toVar();

      const sep = vec3(0).toVar();
      const aliVel = vec3(0).toVar();
      const cohPos = vec3(0).toVar();
      const aliCount = float(0).toVar();
      const cohCount = float(0).toVar();

      Loop(COUNT, ({ i: j }) => {
        const otherPos = positionBuffer.element(j);
        const otherVel = velocityBuffer.element(j);
        const diff = pos.sub(otherPos);
        const dist = length(diff);
        If(dist.greaterThan(0.0001), () => {
          If(dist.lessThan(SEP_DIST), () => {
            sep.addAssign(diff.div(dist).div(dist));
          });
          If(dist.lessThan(ALI_DIST), () => {
            aliVel.addAssign(otherVel);
            aliCount.addAssign(1);
          });
          If(dist.lessThan(COH_DIST), () => {
            cohPos.addAssign(otherPos);
            cohCount.addAssign(1);
          });
        });
      });

      const acc = vec3(0).toVar();
      acc.addAssign(sep.mul(SEP_W));
      If(aliCount.greaterThan(0), () => {
        acc.addAssign(aliVel.div(aliCount).sub(vel).mul(ALI_W));
      });
      If(cohCount.greaterThan(0), () => {
        acc.addAssign(cohPos.div(cohCount).sub(pos).mul(COH_W));
      });

      // Soft spherical boundary.
      const toCenter = center.sub(pos);
      const cdist = length(toCenter);
      If(cdist.greaterThan(BOUND_R), () => {
        acc.addAssign(toCenter.div(cdist).mul(BOUND_W));
      });

      // Shared wind: a steady drift + a gentle swirl around the center so the
      // whole flock wheels like a real murmuration rather than sitting still.
      acc.addAssign(windVec.mul(WIND_GAIN));
      const tangent = cross(vec3(0, 1, 0), toCenter);
      acc.addAssign(normalize(tangent).mul(SWIRL_W));

      // Predator (mouse hawk): flee, falling off with distance.
      const fromPred = pos.sub(predator);
      const pdist = length(fromPred);
      If(pdist.lessThan(PRED_RADIUS), () => {
        const push = float(1.0).sub(pdist.div(PRED_RADIUS));
        acc.addAssign(fromPred.div(max(pdist, 0.0001)).mul(PRED_STRENGTH).mul(push));
      });

      const dt = max(deltaTime, 0.0).min(0.05);
      vel.addAssign(acc.mul(dt));

      const sp = length(vel);
      If(sp.greaterThan(MAX_SPEED), () => {
        vel.assign(vel.div(sp).mul(MAX_SPEED));
      });
      If(sp.lessThan(MIN_SPEED), () => {
        vel.assign(vel.div(max(sp, 0.0001)).mul(MIN_SPEED));
      });

      pos.addAssign(vel.mul(dt));

      // Hard ground floor: push up and reflect downward velocity so birds never
      // sink through the grass.
      If(pos.y.lessThan(FLOOR_Y), () => {
        pos.assign(vec3(pos.x, float(FLOOR_Y), pos.z));
        If(vel.y.lessThan(0.0), () => {
          vel.assign(vec3(vel.x, vel.y.negate().mul(0.5), vel.z));
        });
      });

      positionBuffer.element(i).assign(pos);
      velocityBuffer.element(i).assign(vel);
    })().compute(COUNT);

    const material = new MeshStandardNodeMaterial();
    material.side = DoubleSide;
    material.roughness = 0.8;
    material.metalness = 0.0;

    material.positionNode = Fn(() => {
      const birdPos = positionBuffer.element(instanceIndex);
      const vel = velocityBuffer.element(instanceIndex);

      const fwd = normalize(vel);
      const right = normalize(cross(vec3(0, 1, 0), fwd));
      const up = cross(fwd, right);

      // Flap only the wing tips (aWing=1), so the body stays rigid.
      const phase = hash(instanceIndex).mul(6.2831);
      const flap = sin(time.mul(FLAP_SPEED).add(phase)).mul(FLAP_AMT);
      const wing = attribute("aWing");
      const lx = positionLocal.x;
      const ly = positionLocal.y.add(wing.mul(flap).mul(positionLocal.x.abs()));
      const lz = positionLocal.z;

      return birdPos.add(right.mul(lx)).add(up.mul(ly)).add(fwd.mul(lz));
    })();

    material.colorNode = mix(
      vec3(0.04, 0.05, 0.07),
      vec3(0.16, 0.13, 0.12),
      float(0.4)
    );

    return { material, initKernel, updateKernel, center, windVec, predator };
  }, []);

  const ref = useRef<Object3D>(null!);
  const inited = useRef(false);
  const pointerActive = useRef(false);

  // Track real mouse movement so the predator only engages once the user moves
  // the cursor (otherwise it would sit at screen-center scattering the flock).
  useEffect(() => {
    const onMove = () => (pointerActive.current = true);
    const onLeave = () => (pointerActive.current = false);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerleave", onLeave);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerleave", onLeave);
    };
  }, []);

  // InstancedMesh matrices default to all-zero (which collapses the birds); set
  // them to identity since position is driven entirely by the buffer.
  useEffect(() => {
    const mesh = ref.current as unknown as {
      setMatrixAt: (i: number, m: Object3D["matrix"]) => void;
      instanceMatrix: { needsUpdate: boolean };
    };
    if (!mesh) return;
    const dummy = new Object3D();
    dummy.updateMatrix();
    for (let i = 0; i < COUNT; i++) mesh.setMatrixAt(i, dummy.matrix);
    mesh.instanceMatrix.needsUpdate = true;
  }, []);

  // Wind drift direction/strength shared from the scene controls.
  useEffect(() => {
    const r = (windAngle * Math.PI) / 180;
    sim.windVec.value.set(Math.cos(r) * windStrength, 0, Math.sin(r) * windStrength);
  }, [windAngle, windStrength, sim]);

  useEffect(() => {
    gl.computeAsync(sim.initKernel);
    inited.current = true;
  }, [gl, sim]);

  const plane = useMemo(() => new Plane(new Vector3(0, 1, 0), -CENTER[1]), []);
  const hit = useMemo(() => new Vector3(), []);

  useFrame(() => {
    if (!inited.current) return;

    if (pointerActive.current) {
      raycaster.setFromCamera(pointer, camera);
      if (raycaster.ray.intersectPlane(plane, hit)) {
        sim.predator.value.set(hit.x, hit.y, hit.z);
      }
    } else {
      sim.predator.value.set(FAR_AWAY, FAR_AWAY, FAR_AWAY);
    }

    gl.compute(sim.updateKernel);
  });

  return (
    <instancedMesh
      ref={ref as never}
      args={[geometry, sim.material, COUNT]}
      frustumCulled={false}
    />
  );
}
