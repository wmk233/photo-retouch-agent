const categories = {
  skin: {
    title: "皮肤质感",
    tools: [
      { id: "smooth", label: "光滑肌肤", short: "滑", value: 24, hint: "保留纹理" },
      { id: "skinTone", label: "肤色均匀", short: "肤", value: 12, hint: "校正暗沉" },
      { id: "whiten", label: "自然美白", short: "白", value: 0, hint: "不过曝" },
      { id: "sculpt", label: "立体光影", short: "立", value: 8, hint: "增强轮廓" },
      { id: "acne", label: "面部祛痘", short: "痘", value: 0, hint: "局部净肤" },
      { id: "wrinkle", label: "淡化细纹", short: "纹", value: 0, hint: "保留表情" },
    ],
  },
  face: {
    title: "脸型轮廓",
    tools: [
      { id: "slimFace", label: "瘦脸", short: "瘦", value: 0, hint: "自然收窄" },
      { id: "smallFace", label: "小脸", short: "小", value: 0, hint: "整体缩小" },
      { id: "jawline", label: "下颌线", short: "颌", value: 0, hint: "轮廓清晰" },
      { id: "chin", label: "下巴", short: "下", value: 0, hint: "长度微调" },
      { id: "cheekbone", label: "颧骨", short: "颧", value: 0, hint: "柔和轮廓" },
      { id: "forehead", label: "额头", short: "额", value: 0, hint: "比例微调" },
      { id: "doubleChin", label: "淡化双下巴", short: "颈", value: 0, hint: "收紧轮廓" },
    ],
  },
  eyes: {
    title: "眼部精修",
    tools: [
      { id: "enlargeEyes", label: "放大眼睛", short: "大", value: 0, hint: "双眼同步" },
      { id: "eyeBags", label: "去除眼袋", short: "袋", value: 0, hint: "保留卧蚕" },
      { id: "darkCircles", label: "淡化黑眼圈", short: "黑", value: 0, hint: "自然提亮" },
      { id: "iris", label: "眼眸增强", short: "眸", value: 0, hint: "增强细节" },
      { id: "brightEyes", label: "亮眼", short: "亮", value: 0, hint: "清澈有神" },
      { id: "eyeDistance", label: "眼距", short: "距", value: 0, hint: "比例微调" },
      { id: "eyeAngle", label: "眼角", short: "角", value: 0, hint: "轻微调整" },
    ],
  },
  features: {
    title: "五官细节",
    tools: [
      { id: "noseBridge", label: "立体鼻梁", short: "鼻", value: 0, hint: "自然高光" },
      { id: "slimNose", label: "瘦鼻", short: "窄", value: 0, hint: "鼻翼收窄" },
      { id: "teeth", label: "牙齿美白", short: "牙", value: 0, hint: "中性白" },
      { id: "smile", label: "自然微笑", short: "笑", value: 0, hint: "轻调表情" },
      { id: "lipVolume", label: "唇形", short: "唇", value: 0, hint: "丰盈度" },
      { id: "lipColor", label: "唇色", short: "色", value: 0, hint: "自然气色" },
      { id: "brows", label: "眉形", short: "眉", value: 0, hint: "提升精神" },
    ],
  },
  body: {
    title: "身形塑造",
    tools: [
      { id: "slimBelly", label: "瘦肚子", short: "腹", value: 0, hint: "收紧腹部" },
      { id: "slimWaist", label: "细腰", short: "腰", value: 0, hint: "保护背景" },
      { id: "slimArms", label: "瘦手臂", short: "臂", value: 0, hint: "双侧联动" },
      { id: "slimLegs", label: "瘦腿", short: "腿", value: 0, hint: "自然线条" },
      { id: "longLegs", label: "长腿", short: "长", value: 0, hint: "比例优化" },
      { id: "shoulders", label: "肩宽", short: "肩", value: 0, hint: "轮廓微调" },
      { id: "hips", label: "提臀", short: "臀", value: 0, hint: "自然曲线" },
      { id: "headRatio", label: "头身比", short: "比", value: 0, hint: "整体比例" },
    ],
  },
  makeup: {
    title: "自然妆容",
    tools: [
      { id: "foundation", label: "底妆", short: "底", value: 0, hint: "轻薄服帖" },
      { id: "blush", label: "腮红", short: "腮", value: 0, hint: "提升气色" },
      { id: "contour", label: "修容", short: "修", value: 0, hint: "自然立体" },
      { id: "eyebrowMakeup", label: "眉妆", short: "眉", value: 0, hint: "柔和填充" },
      { id: "eyeliner", label: "眼线", short: "线", value: 0, hint: "眼神增强" },
      { id: "lipstick", label: "口红", short: "唇", value: 0, hint: "自然色泽" },
    ],
  },
  tone: {
    title: "整体氛围",
    tools: [
      { id: "brightness", label: "亮度", short: "亮", value: 0, hint: "整体曝光" },
      { id: "contrast", label: "对比度", short: "对", value: 0, hint: "明暗层次" },
      { id: "warmth", label: "色温", short: "温", value: 0, hint: "冷暖调节" },
      { id: "saturation", label: "饱和度", short: "饱", value: 0, hint: "色彩浓度" },
      { id: "clarity", label: "清晰度", short: "清", value: 0, hint: "细节增强" },
      { id: "background", label: "背景净化", short: "景", value: 0, hint: "弱化干扰" },
    ],
  },
};

function createStore(initial) {
  const listeners = new Map();

  const state = new Proxy(initial, {
    set(target, key, value) {
      target[key] = value;
      (listeners.get(key) || []).forEach((fn) => fn(value));
      (listeners.get("*") || []).forEach((fn) => fn(key, value));
      return true;
    },
  });

  return {
    state,
    on(key, fn) {
      if (!listeners.has(key)) listeners.set(key, []);
      listeners.get(key).push(fn);
    },
  };
}

const initialValues = Object.fromEntries(
  Object.values(categories)
    .flatMap((cat) => cat.tools)
    .map((tool) => [tool.id, tool.value]),
);

export const { state, on } = createStore({
  category: "skin",
  mode: "local",
  values: initialValues,
  history: [],
  selectedPreset: "natural",
  originalOnly: false,
  sourceFile: null,
  sourceName: "",
  capabilities: { brainProviders: [], actionProviders: [] },
  aiResultImage: null,
  aiResultUrl: "",
  displayingAiResult: false,
  aiBusy: false,
});

export function currentTools(searchQuery = "") {
  const query = searchQuery.toLowerCase();
  const tools = categories[state.category].tools;
  if (!query) return tools;
  return tools.filter(
    (tool) => `${tool.label}${tool.hint}`.toLowerCase().includes(query),
  );
}

export const allTools = Object.values(categories).flatMap((cat) => cat.tools);

export const toolMap = Object.fromEntries(allTools.map((t) => [t.id, t]));

export { categories };
