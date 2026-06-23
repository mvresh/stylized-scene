import {
  abs,
  cameraViewMatrix,
  normalize,
  normalWorld,
  vec3,
  vec4,
} from "three/tsl";

// --- Double-sided lighting fix ---
// With DoubleSide, three.js flips the shading normal on back-facing fragments.
// On an up-pointing grass blade or an outward-facing leaf card that rotates the
// normal down into the dark lower hemisphere, so the back face reads as unlit
// even though the albedo is correct. Mirror the world normal back into the
// upper (sky-lit) hemisphere by forcing a positive world-up component, so both
// faces sample the same sky/ambient light while keeping the sideways shaping
// from the surface's outward normal.
//
// Returns both spaces: `view` is ready to assign to material.normalNode, while
// `world` is exposed so callers (grass) can reuse it for view-dependent terms
// like translucency and fresnel without recomputing it.
export function skyHemisphereNormal() {
  const world = normalize(
    vec3(normalWorld.x, abs(normalWorld.y), normalWorld.z)
  );
  const view = normalize(cameraViewMatrix.mul(vec4(world, 0)).xyz);
  return { world, view };
}
