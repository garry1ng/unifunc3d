# Attribution and third-party notices

- The evaluation protocol follows [Fun3DU](https://github.com/tev-fbk/fun3du). `evaluate.py` and `release_lib/common.py` implement its point precision/recall threshold metrics for sparse predictions.
- Evaluation uses the [SceneFun3D benchmark](https://github.com/SceneFun3D/scenefun3d). The upstream MIT notice is retained in [licenses/SceneFun3D-MIT.txt](licenses/SceneFun3D-MIT.txt). Dataset assets must be obtained separately and remain subject to the [upstream terms](https://scenefun3d.github.io/documentation/license/).
- The released predictions were produced by UniFunc3D using [Qwen3](https://github.com/QwenLM/Qwen3) and [SAM3](https://github.com/facebookresearch/sam3). Model code and weights are not included in this initial release.
- Prediction JSONs contain derived point selections, without raw scans, images, or ground-truth masks. The figures in `assets/` are from the [UniFunc3D project page](https://jiaying.link/unifunc3d/).

Existing authorship, copyright, and upstream terms are preserved. This notice does not grant a new license over third-party or project-specific components.
