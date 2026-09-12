/**
 * Paper-backed content and external-resource configuration.
 * Empty link strings are disabled automatically by the UI.
 */
window.PROJECT_DATA = {
  shortName: "FOM-SAM3-Policy",
  title: "Towards Fine-Grained Object Manipulation: SAM3-Guided Visuomotor Policy with Persistent Memory Learning and Focused Visual Conditioning",
  label: "FOM-SAM3-Policy · Project Page",
  authors: [
    "Haolong Meng¹˒²",
    "Fangbo Qin*¹˒²",
    "Mengchen Bai¹",
    "Houwu Wang¹",
    "Cirong Liu³",
    "Shan Yu¹˒²",
  ],
  institutions: [
    "¹ Institute of Automation, Chinese Academy of Sciences",
    "² School of Artificial Intelligence, University of Chinese Academy of Sciences",
    "³ Center for Excellence in Brain Science and Intelligence Technology, Chinese Academy of Sciences",
  ],
  summary:
    "FO Memory gives frozen SAM3 a persistent fine-grained identity, while Focused Spatial-Appearance Encoding turns the selected target and its task context into policy-ready conditions for DP or ACT.",
  abstract: [
    "Imitation learning of visuomotor policies endows the robot with dexterous manipulation skills. However, even in a familiar scene where a robot repeats its tasks, distinguishing a target fine-grained object (FO) from similar objects in the same coarse category remains an unresolved issue. Besides, current visuomotor policies encode the scene image as the visual condition, which might be prone to visual distractors. To address these problems, we first propose FO Memory-enhanced Segment Anything Model 3 (FOM-SAM3). With SAM3 fully frozen, it converts a coarse concept prompt into reusable FO Memory Tokens from a few multi-view registration images. At deployment, the tokens directly prompt frozen SAM3 without registration images or model adaptation. Second, we introduce Focused Spatial-Appearance Encoding (FSAE), which focuses the action policy on ordered target and manipulation-context regions and models their local appearance and relative geometry. FSAE produces a compact condition vector for DP or object condition tokens for ACT without changing either action objective. Experiments on 30 FOs from 4 coarse categories show that FO Memory improves target segmentation, ranks the queried FO above visually similar candidates, suppresses Similar-FO confusion, and rejects absent identities more reliably. Real-robot experiments report completed and failed rollouts and evaluate within-type new-FO reuse with frozen policies.",
  ],
  abstractHighlights: [
    "fine-grained object (FO)",
    "FO Memory-enhanced Segment Anything Model 3 (FOM-SAM3)",
    "reusable FO Memory Tokens",
    "Focused Spatial-Appearance Encoding (FSAE)",
    "30 FOs from 4 coarse categories",
    "Real-robot experiments",
  ],
  links: {
    paper: "",
    arxiv: "",
    code: "",
    video: "#demonstrations-title",
  },
  demoVideos: [
    { src: "assets/videos/1.mp4", title: "Collect Can · Continuous long-horizon rollout" },
    { src: "assets/videos/2.mp4", title: "Push Box · Continuous long-horizon rollout" },
    { src: "assets/videos/3.mp4", title: "Lid Cup · Continuous long-horizon rollout" },
    { src: "assets/videos/gen4.mp4", title: "Collect Can · Blue beverage-can FO" },
    { src: "assets/videos/gen5.mp4", title: "Push Box · White cylindrical FO" },
    { src: "assets/videos/gen6.mp4", title: "Push Box · Green cylindrical FO" },
    { src: "assets/videos/gen7.mp4", title: "Lid Cup · Stainless-steel lid FO" },
    { src: "assets/videos/gen8.mp4", title: "Lid Cup · Decorated lid FO" },
    { src: "assets/videos/gen9.mp4", title: "Lid Cup · New-background rollout" },
    { src: "assets/videos/gen10.mp4", title: "Push Box · Cube-shaped FO" },
    { src: "assets/videos/gen11.mp4", title: "Push Box · Blue carton FO" },
  ],
  generalizationDemos: [
    { task: "Collect Can", taskKey: "pick_place", method: "Ours-DP", methodKey: "ours_dp", sample: "01" },
    { task: "Collect Can", taskKey: "pick_place", method: "Ours-DP", methodKey: "ours_dp", sample: "02" },
    { task: "Collect Can", taskKey: "pick_place", method: "Ours-ACT", methodKey: "ours_act", sample: "01" },
    { task: "Collect Can", taskKey: "pick_place", method: "Ours-ACT", methodKey: "ours_act", sample: "02" },
    { task: "Push Box", taskKey: "push", method: "Ours-DP", methodKey: "ours_dp", sample: "01" },
    { task: "Push Box", taskKey: "push", method: "Ours-DP", methodKey: "ours_dp", sample: "02" },
    { task: "Push Box", taskKey: "push", method: "Ours-ACT", methodKey: "ours_act", sample: "01" },
    { task: "Push Box", taskKey: "push", method: "Ours-ACT", methodKey: "ours_act", sample: "02" },
    { task: "Lid Cup", taskKey: "assemble", method: "Ours-DP", methodKey: "ours_dp", sample: "01" },
    { task: "Lid Cup", taskKey: "assemble", method: "Ours-DP", methodKey: "ours_dp", sample: "02" },
    { task: "Lid Cup", taskKey: "assemble", method: "Ours-ACT", methodKey: "ours_act", sample: "01" },
    { task: "Lid Cup", taskKey: "assemble", method: "Ours-ACT", methodKey: "ours_act", sample: "02" },
  ],
  taskPrompts: {
    "Pick & Place": "red CocaCola soda can",
    Push: "white pill box",
    Assemble: "cup lid with panda handle",
  },
  robotResults: [
    { method: "RGB-DP", id: ["8/10", "7/10", "5/10"], distractors: ["6/10", "3/10", "5/10"], similar: ["1/10", "3/10", "0/10"] },
    { method: "S²-Diff.", id: ["10/10", "7/10", "4/10"], distractors: ["8/10", "6/10", "5/10"], similar: ["4/10", "6/10", "5/10"] },
    { method: "Ours-DP", id: ["10/10", "9/10", "8/10"], distractors: ["10/10", "8/10", "10/10"], similar: ["10/10", "10/10", "10/10"], ours: true },
    { method: "RGB-ACT", id: ["10/10", "8/10", "6/10"], distractors: ["4/10", "5/10", "3/10"], similar: ["0/10", "5/10", "1/10"] },
    { method: "Ours-ACT", id: ["10/10", "9/10", "9/10"], distractors: ["8/10", "10/10", "9/10"], similar: ["8/10", "9/10", "7/10"], ours: true },
  ],
};
