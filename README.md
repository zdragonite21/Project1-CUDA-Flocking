**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

- Zachary Leong
  - [LinkedIn](https://linkedin.com/in/zleong), [personal website](https://zacharyleong.com)
- Tested on: Windows 11, Ultra 9 185H @ 2.30GHz 32GB, RTX 4060 Laptop (personal)

<p align="center">
  <video controls src="boids_mystic_torus.mp4" title="Title"></video>
  <i>1 million boids</i>
</p>

### pull request

- implemented boids algorithm (3 rules) using CUDA
- 3 implementations: naive, uniform grid, **coherent** grid
- extra credit: grid-looping optimization

## boids

boids are particles that independently follow 3 rules:

1. cohesion
2. separation
3. alignment

And from there, emerges complex and beautiful behavior, resembling flocks of birds, or schools of fish.

### sdf forces

Inspired by Sebastian Lague's [video](https://www.youtube.com/watch?v=bqtqltqcQhw) about boids, I initially attempted to make the boids avoid collisions with objects and the bounds of the simulation (currently they teleport when they are outside of the bounds). However, having each boid perform a number of raycasts to choose a unobstructed direction would be infeasible on the gpu without an acceleartion structure.

Instead, I went with the simpler method: use SDFs and their gradients to influence the boids' velocities. Using the box SDF formula from Inigo Quilez's [blog](https://iquilezles.org/articles/distgradfunctions3d/), I nudged the velocities of boids near the bounds to stay inside.

<video controls src="boids_avoidance.mp4" title="Title"></video>

This is how it works:

An SDF (`sdf(vec3 p)`) tells us the signed distance from the point `p` and some mathematical representation of a shape.

- `p` is inside the shape: `sdf(vec3 p) < 0`
- `p` is outside the shape: `sdf(vec3 p) > 0`
- `p` is on the surface of the shape: `sdf(vec3 p) = 0`

Along with the distance, for certain sdf shapes, there are analytical gradients (`gdf(p)`) that can be computed given a `p`.

For example, distance and gradient together:

```c
// IQ's blog, 3D distance and gradient functions - 2025
vec4 sdgBox( in vec3 p, in vec3 b, in float r )
{
    vec3  w = abs(p)-(b-r);
    float g = max(w.x,max(w.y,w.z));
    vec3  q = max(w,0.0);
    float l = length(q);
    vec4  f = (g>0.0)?vec4(l, q/l) :
                      vec4(g, w.x==g?1.0:0.0,
                              w.y==g?1.0:0.0,
                              w.z==g?1.0:0.0);
    return vec4(f.x-r, f.yzw*sign(p));
}
```

---

This function returns a `vec4` where `x` is the signed distance, and `yzw` is the gradient.

The gradient tells us the direction in which the SDF function increases the fastest, which is also known as the **normal**. If a boid is within a certain distance to our boundary box, we can apply a force in the direction of the _negative_ gradient (since we are inside the box). In otherwords, we move away from the surface if we are too close to it.

Now we can also use the SDF as an attractive force, creating effects like this.

<p align="center">
  <video controls src="boids_covering.mp4" title="Title"></video>
  <i>torus sdf</i>
</p>

<p align="center">
  <video controls src="mandelbulb.mp4" title="Title"></video>
  <i>mandelbulb sdf</i>
</p>

<p align="center">
  <video controls src="mandelbulb_attraction.mp4" title="Title"></video>
  <i>mandelbulb attraction</i>
</p>

### performance analysis

![alt text](<Average FPS vs Block Size (100k boids, avg over 5s, viz off).png>)

![alt text](<Average FPS vs Boid Count (128 Block Size, Viz Off, Avg Over 10s).png>)

![alt text](<Average FPS vs Cell Width (100k boids, avg over 5s, coherent grid, 128 block size, viz off).png>)

### cmake lists modification

I moved `include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")` so that the CUDA toolkit is included along with Windows builds as CMake Tools in VS Code doesn't automatically include this path like Visual Studio does.

I also added a compile option for release mode to include the `-lineinfo` tag for NSight Compute performance analysis.

Added: `$<COMPILE_LANGUAGE:CUDA>>:-G>" "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>"`
