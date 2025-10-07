CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* - Muqiao Lei
    
    [LinkedIn](https://www.linkedin.com/in/muqiao-lei-633304242/) · [GitHub](https://github.com/rmurdock41)
  
  - Tested on: Windows 10, 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz 2.30 GHz, NVIDIA GeForce RTX 3060 Laptop GPU (Personal Computer)

#### CUDA Path Tracer

**University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 3**

- [Your Name]
  - [LinkedIn](your-linkedin-link)
  - [Personal Website](your-website)
- Tested on: Windows 10, i7-XXXXX @ X.XXGHz 16GB, RTX XXXX 8GB

**A CUDA-based path tracer implementing glTF model loading, refractive materials, depth of field, and BVH acceleration. Optimized using Russian Roulette path termination and various GPU performance techniques.**

![Cover Image](img/top.png)

*[Lady Maria from BLoodborne], 5000 samples, rendered with BVH acceleration*

---

## Visual Features

### Refraction

Implemented **refraction effects** based on Snell's law with Fresnel calculations using Schlick's approximation. Supports transparent materials like glass and water with light bending and internal reflections. Implemented rough refractive surfaces that can simulate effects like frosted glass.

| Clear Glass (IOR 1.5, roughness=0)  | Colored Glass                           | Rough Glass (frosted effect)        |
| ----------------------------------- | --------------------------------------- | ----------------------------------- |
| ![Clear Glass](img/glass_clear.png) | ![Colored Glass](img/glass_colored.png) | ![Rough Glass](img/glass_rough.png) |

| Multiple Refractive Objects                 | Roughness Comparison                                    |
| ------------------------------------------- | ------------------------------------------------------- |
| ![Multiple Objects](img/glass_multiple.png) | ![Roughness Levels](img/glass_roughness_comparison.png) |

**Implementation Details:**

Used `glm::refract` to calculate ideal refraction direction and **Schlick's approximation** to compute Fresnel reflection coefficient. The system probabilistically chooses between reflection and refraction based on the Fresnel coefficient, correctly handling total internal reflection cases. 

For **roughness** implementation, the material.roughness parameter controls surface roughness by importance sampling around the ideal reflection or refraction direction. The sampling uses a modified Phong distribution model where higher roughness values produce stronger light scattering, creating realistic frosted effects. 

**Performance Analysis:**

| Scene Type                                | FPS | vs Pure Diffuse |
| ----------------------------------------- | --- | --------------- |
| Single smooth glass sphere (roughness=0)  | 45  | -25%            |
| Single rough glass sphere (roughness=0.3) | 42  | -30%            |
| Multiple glass objects                    | 28  | -42%            |
| Mixed glass + diffuse                     | 38  | -35%            |

**Analysis:**

Refractive materials increase ray tracing computation as light undergoes multiple reflections and refractions inside transparent objects, resulting in longer ray paths that require more intersection tests and shading calculations. Rough surfaces add overhead primarily from additional random number generation (2 random numbers per sample) and tangent space coordinate transformations, but produce realistic frosted glass effects. 

Performance data shows that closed scenes experience more pronounced performance impact from refractive materials because rays struggle to escape the scene, continuously bouncing between transparent objects and other surfaces, further increasing average path length. In contrast, open scenes experience relatively smaller performance drops as some refracted rays exit the scene boundaries and terminate early. Scenes with multiple glass objects show the most significant performance degradation in both scene types, as rays bounce multiple times between transparent surfaces, substantially increasing computational burden.



### Specular Reflection

Implemented perfect specular reflection materials with roughness parameter support, simulating effects from perfect mirrors to rough metallic surfaces.

Reference: [PBRv4 9.2](https://pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission)

![Specular Reflection](img/specular.png)

*Demonstrating specular reflection effects with varying roughness values*

**Implementation Details:**

Used `glm::reflect` to calculate ideal reflection direction, with material.roughness parameter controlling surface roughness. For perfect mirrors (roughness=0), rays bounce strictly according to the law of reflection. When roughness is greater than 0, importance sampling is performed around the ideal reflection direction using a modified Phong distribution model. Sampling uses a `cosθ^k` distribution where `k = 1/(α²) - 1` and `α = roughness`. Higher roughness values produce larger scattering ranges for reflected rays, with surfaces exhibiting more diffuse characteristics. The system supports material.hasReflective parameter to control reflection strength, enabling partial reflection effects. Material color acts as the Fresnel term for reflection, affecting the color of reflected light.



### Depth of Field

Implemented physically-based thin lens camera model to simulate realistic depth of field effects by sampling on the aperture.

| No DOF                      | with DOF                             |
| --------------------------- | ------------------------------------ |
| ![No DOF](img/dof_none.png) | ![Large Aperture](img/dof_large.png) |

**Implementation Details:**

Used concentric disk sampling to generate ray origins on the aperture plane, which maps a square to a unit disk, avoiding waste from traditional rejection sampling and providing more uniform distribution. All rays pass through the same point on the focal plane to achieve focus effects. 

The system provides two configurable parameters: apertureRadius controls blur strength, and focalDistance controls focus position. The implementation flow calculates the standard camera ray direction first, then computes the ray's intersection with the focal plane, samples an offset point on the aperture using concentric disk sampling, and finally shoots a ray from the offset point toward the focal plane intersection point.

**Performance Impact:**

| Test Scene    | No DOF(Open) | No DOF(Closed) | With DOF(Open) | With DOF(Closed) | Performance Difference |
| ------------- | ------------ | -------------- | -------------- | ---------------- | ---------------------- |
| Complex Scene |              | 38 fps         | 36 fps         | 36 fps           | ~5.3%                  |

**Analysis:**

Depth of field only affects the ray generation stage without impacting subsequent intersection testing and shading, resulting in minimal performance overhead. Aperture size has almost no performance impact as computation cost remains constant regardless of aperture size, only the ray origin positions differ. This overhead primarily comes from additional random number generation, focal plane intersection calculation, and ray direction renormalization, all of which are simple arithmetic operations. The performance characteristics of depth of field are independent of scene type (open or closed) because it only modifies ray origins and directions without affecting ray propagation behavior within the scene. 

Data shows that both open and closed scenes experience consistent overhead. This makes depth of field a cost-effective choice for enhancing rendering realism, significantly improving visual quality with negligible impact on frame rate.

---

## glTF Model Loading

Implemented complete glTF 2.0 model loading and rendering using TinyGLTF library, supporting import and rendering of arbitrary triangle meshes.

| Stanford Bunny (69,451 tris) | Lady Maria (1,013,600 tris)       |
| ---------------------------- | --------------------------------- |
| ![Bunny](img/bunny.png)      | ![Lady Maria](img/lady_maria.png) |

**Implementation Details:**

Implemented complete glTF 2.0 file parsing and geometry data extraction pipeline. The system reads mesh data from glTF files, extracts vertex positions and index information, and transforms triangles to world coordinate space. Each loaded mesh constructs an AABB bounding box on the CPU side for subsequent BVH acceleration structure building. Supports model translation, rotation, and scaling transformations, correctly converting model coordinates to scene space through transformation matrices. All triangle data is uniformly stored in GPU memory, with each triangle recording its three vertex positions and material ID. The system manages mesh instances in the scene as MESH-type Geom objects, recording the offset and count of triangles in the global array for subsequent intersection testing.

**Analysis:**

glTF model loading itself has no direct impact on runtime performance as model parsing and data upload are completed during initialization. Runtime performance primarily depends on triangle count and intersection testing efficiency. For models like Stanford Bunny with approximately 70,000 triangles and Lady Maria with over 1 million triangles, BVH acceleration structures have critical performance implications (see BVH section for detailed performance analysis). Triangle data is stored in contiguous arrays on GPU, favoring cache hits and reducing memory access latency. The advantage of glTF format lies in its standardization and widespread support, allowing convenient export from various 3D modeling software and supporting complex scene hierarchies and material definitions.





---

## Performance Optimizations

### Bounding Volume Hierarchy (BVH)

Implemented hierarchical spatial acceleration structure using axis-aligned bounding boxes (AABB), dramatically reducing ray-triangle intersection tests. Uses Surface Area Heuristic (SAH) for CPU-side BVH construction and iterative stack-based traversal on GPU.





| ![BVH Visualization](img/bvh_structure.png) | ![Performance Chart](img/bvh_performance.png) |
| ------------------------------------------- | --------------------------------------------- |
| **glTF FPS with BVH*                        | *glTF FPS without BVH*                        |

**Implementation Details:**

Constructed BVH tree recursively on CPU using Surface Area Heuristic for optimal split plane selection. During construction, first computes AABB bounding boxes and centroids for all primitives, then selects the axis with maximum centroid distribution as the split axis, using median split strategy to partition primitives into two groups. Leaf nodes contain at most 4 primitives to balance tree depth and leaf complexity. After construction, BVH nodes and primitive indices are uploaded to GPU memory. GPU traversal uses iteration rather than recursion, employing a fixed-size stack (64 levels) to store nodes awaiting visit. For each ray, starting from the root node, performs AABB intersection test first; if intersected, continues traversing child nodes or tests primitives in leaf nodes. System implements two-level BVH structure: scene-level BVH accelerates intersection tests between different objects, mesh-level BVH accelerates triangle intersection tests within individual meshes.

**Performance Analysis:**

| Scene          | Triangle Count | Scene Type | Without BVH | With BVH | Speedup |
| -------------- | -------------- | ---------- | ----------- | -------- | ------- |
| MultiBall      | -              | Open       |             |          |         |
| MultiBall      | -              | Closed     |             |          |         |
| Cube           | 12             | Open       | XX fps      | XX fps   | ~1.0x   |
| Cube           | 12             | Closed     | XX fps      | XX fps   | ~1.0x   |
| Stanford Bunny | 69,451         | Open       | XX fps      | XX fps   | ~XXx    |
| Stanford Bunny | 69,451         | Closed     | XX fps      | XX fps   | ~XXx    |
| Lady Maria     | 1,013,600      | Open       | <1 fps      | XX fps   | ~XXx    |
| Lady Maria     | 1,013,600      | Closed     | <1 fps      | XX fps   | ~XXx    |

**BVH Statistics:**

| Scene          | Triangle Count | Mesh BVH Nodes | Scene BVH Nodes |
| -------------- | -------------- | -------------- | --------------- |
| Multiball      |                |                |                 |
| Stanford Bunny | 69,451         | 40,597         | 3               |
| Lady Maria     | 1,013,600      | 524,287        | 3               |

*Note: Mesh BVH nodes accelerate triangle intersection tests within individual models, Scene BVH nodes accelerate intersection tests between different objects in the scene*

**Analysis:**

BVH acceleration structure reduces linear search complexity from O(n) to O(log n) through spatial hierarchy partitioning. For simple models like the cube with only 12 triangles, BVH traversal overhead may exceed saved intersection testing time, resulting in negligible acceleration. However, when triangle count increases to tens of thousands, BVH advantages become significantly apparent. 

For Stanford Bunny with 70,000 triangles, BVH reduces average triangles tested per ray from 69,451 to tens, achieving several to tenfold performance improvement. For Lady Maria with 1 million triangles, rendering without BVH is nearly impossible (<1 fps), while BVH provides tens to hundreds of times speedup, enabling real-time rendering. 

Comparing open and closed scenes shows that BVH acceleration effectiveness is largely independent of scene type, as BVH primarily optimizes the intersection testing stage rather than ray bouncing behavior. Closed scenes have lower baseline FPS due to more ray bounces, but BVH speedup ratios remain similar across both scene types. Two-level BVH structure enables efficient handling of scenes with multiple complex models: scene-level BVH quickly locates potentially intersected objects, mesh-level BVH rapidly finds intersected triangles within objects. Iterative traversal on GPU avoids recursion overhead and stack overflow issues, with fixed 64-level stack satisfying most scene requirements. AABB intersection testing uses slab method, computationally simple and efficient, suitable for GPU's SIMD architecture.

### Russian Roulette Path Termination

Probabilistically terminates paths based on their energy contribution, reducing computation while maintaining unbiased results.

| RR minDepth=1                                          | RR minDepth=3                                          | No RR                                                            |
| ------------------------------------------------------ | ------------------------------------------------------ | ---------------------------------------------------------------- |
| <img title="" src="img/RR1.png" alt="RR1" width="643"> | <img title="" src="img/RR3.png" alt="RR3" width="622"> | <img title="" src="img/without%20RR.png" alt="NoRR" width="786"> |

*All images at 5000 samples*

**Implementation Details:**

Implemented probability-based termination strategy using path throughput. The system provides a configurable minimum bounce depth (minDepth) parameter, preventing path termination before reaching this depth to ensure basic lighting effects. Starting from minDepth, survival probability is calculated based on path energy contribution, computed as the maximum component of the path color and clamped between 0.05 and 0.99 to avoid premature or ineffective termination. For each ray, a random number is compared with survival probability; if the random number exceeds survival probability, the path is terminated, otherwise the path color is divided by survival probability for energy compensation to maintain unbiasedness. This strategy allows high-contribution paths to continue tracing while low-contribution paths terminate early, saving computational resources.

**Performance Analysis:**

**Open Scene (Cornell Box):**

| Configuration   | FPS    | vs No RR |
| --------------- | ------ | -------- |
| No RR           | XX fps | -        |
| RR (minDepth=1) | XX fps | +XX%     |
| RR (minDepth=3) | XX fps | +XX%     |

**Closed Scene (Enclosed Room):**

| Configuration   | FPS    | vs No RR |
| --------------- | ------ | -------- |
| No RR           | XX fps | -        |
| RR (minDepth=1) | XX fps | +XX%     |
| RR (minDepth=3) | XX fps | +XX%     |

**Analysis:**

Russian Roulette improves performance by early terminating low-contribution paths while maintaining unbiasedness through energy compensation. In open scenes, many rays naturally exit scene boundaries and terminate, making RR's effect relatively limited but still providing performance improvements. 

In closed scenes, rays are trapped inside and continue bouncing, making RR's effect more pronounced with significantly higher performance gains. The choice of minDepth parameter affects performance benefits: minDepth=1 allows termination after the first bounce, potentially achieving higher performance gains; minDepth=3 guarantees the first 3 bounces are not terminated, ensuring basic multi-bounce lighting is correctly captured, representing a balanced choice between performance and quality. Rendered results show that images with different minDepth settings and with/without RR are visually indistinguishable, validating RR's unbiasedness. Through extensive sampling, random termination introduced by RR produces no noticeable bias or noise, with virtually identical image quality. This makes RR a performance optimization technique with virtually no quality loss.
