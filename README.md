# stylized-scene + GPU murmuration

A stylized, Ghibli-inspired grass field rendered with React Three Fiber on a
WebGPU backend — **with a GPU-simulated starling murmuration wheeling above it**.
The grass and foliage are shaded and animated entirely on the GPU with Three.js'
node material system (TSL): an instanced grass field, a tree built from a trunk
model and instanced canopy bushes, and a layered, directional wind that bends
everything in a travelling gust. A flock of birds is simulated on the GPU and
rides the same wind.

> This is a fork of [`dedekpo/stylized-scene`](https://github.com/dedekpo/stylized-scene)
> by Andre Elias. The original is the grass field, wind, ground and tree; this
> fork adds the bird murmuration (`src/scene/flock.tsx`) on top.

## Features

- **WebGPU + TSL** node materials throughout (no GLSL strings).
- **GPU murmuration** — ~1,600 birds simulated entirely on the GPU with TSL
  compute: Reynolds separation / alignment / cohesion, a soft spherical
  boundary, a hard ground floor, a shared wind drift + swirl, and a
  mouse-driven predator. The flock is one instanced body+wings mesh whose
  vertex stage reads the sim buffers, orients each bird along its velocity and
  flaps its wings. See [`src/scene/flock.tsx`](src/scene/flock.tsx).
- **Instanced grass** with per-clump color variation, ground-color projection,
  translucency and a Fresnel rim.
- **Directional wind node** shared by the grass, the tree, and the flock: a
  circular-arc bend driven by a travelling gust wave, breeze, chop and tip
  flutter. See [`src/materials/wind.ts`](src/materials/wind.ts).
- **Layered ground** material blending grass and dirt with a painted path mask.
- A live **controls panel** for tuning colors, wind and debug views.

## Running

```sh
npm install
npm run dev
```

Requires a **WebGPU-capable browser** (recent Chrome, Edge, or Safari).

## Scripts

- `npm run dev` — start the Vite dev server
- `npm run build` — type-check and build for production
- `npm run lint` — run ESLint

## Layout

```
src/
  materials/   # TSL node materials (grass, ground, wind, normals)
  scene/       # R3F components (grass, ground, tree, flock, scene root)
  ui/          # controls panel + state
  utils/       # geometry, texture and uniform helpers
  config/      # scene constants and asset paths
```

## Credits & License

[MIT](LICENSE) © Andre Elias for the original
[stylized-scene](https://github.com/dedekpo/stylized-scene). Murmuration
addition by this fork, released under the same MIT license.
