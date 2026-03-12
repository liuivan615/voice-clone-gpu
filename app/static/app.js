/**
 * Signal Foundry Workstation — app.js
 * 重写版本：修复重复函数 bug、补全缺失函数、新增 Tab 导航、F0 卡片、训练完成引导等
 */

// ── F0 Predictor 元数据 ──────────────────────────────────────────────────────
const PREDICTOR_META = {
  rmvpe:   { title: "RMVPE",   badge: "推荐", tier: "recommended", description: "精度优先，适合歌声、复杂录音和商业级默认方案。资源占用更高，但通常是效果最稳的一档。" },
  fcpe:    { title: "FCPE",    badge: null,   tier: "fast",        description: "更快的现代替代，适合想缩短前处理时间的场景。质量通常接近 RMVPE。" },
  crepe:   { title: "CREPE",   badge: null,   tier: "legacy",      description: "旧神经网络方案，保留做实验和兼容，不作为默认推荐。" },
  harvest: { title: "Harvest", badge: null,   tier: "light",       description: "传统轻量方案，兼容性好，但复杂音频通常不如 RMVPE。" },
  dio:     { title: "DIO",     badge: null,   tier: "light",       description: "更轻的传统方法，适合 smoke test 或 CPU 预处理。" },
  pm:      { title: "PM",      badge: null,   tier: "light",       description: "历史兼容方法，只建议做对比或兜底。" },
};

// 主卡片展示：RMVPE + FCPE；其余折叠
const PREDICTOR_PRIMARY = ["rmvpe", "fcpe"];

const SEPARATOR_META = {
  demucs: {
    title: "Demucs",
    badge: "默认",
    tier: "recommended",
    description: "质量优先，适合伴奏复杂、混响较多的素材，默认推荐作为预处理入口。",
  },
  mdx: {
    title: "MDX-Net",
    badge: null,
    tier: "fast",
    description: "经典人声分离路线，适合做备选对照，速度与效果更偏向平衡。",
  },
};

const TRAINING_PRESET_META = {
  balanced: { icon: "⚡", fallbackLabel: "均衡", accent: "balanced" },
  high_quality: { icon: "🎯", fallbackLabel: "高质量", accent: "quality" },
  light: { icon: "💡", fallbackLabel: "轻量", accent: "light" },
};

// 防御性兜底：仅在后端 overview 未返回 training_presets 时使用。
// 应与 training_preset_service.py build_training_presets() 保持同步。
const FALLBACK_TRAINING_PRESETS = [
  {
    id: "balanced",
    label: "均衡",
    encoder: "vec768l12",
    encoder_dim: 768,
    ssl_dim: 768,
    gin_channels: 768,
    filter_channels: 768,
    recommended_vram_gb: 6,
    description: "推荐大多数场景，质量与速度更平衡。",
    is_default: true,
    available: true,
    reason_disabled: null,
  },
  {
    id: "high_quality",
    label: "高质量",
    encoder: "whisper-ppg",
    encoder_dim: 1024,
    ssl_dim: 1024,
    gin_channels: 1024,
    filter_channels: 1024,
    recommended_vram_gb: 12,
    description: "内容还原更精细，但训练更慢、资源更高。",
    is_default: false,
    available: true,
    reason_disabled: null,
  },
  {
    id: "light",
    label: "轻量",
    encoder: "hubertsoft",
    encoder_dim: 256,
    ssl_dim: 256,
    gin_channels: 256,
    filter_channels: 768,
    recommended_vram_gb: 4,
    description: "适合显存偏紧的机器，优先保证能稳定开跑。",
    is_default: false,
    available: true,
    reason_disabled: null,
  },
];

// 防御性兜底：仅在后端 overview 未返回 training_capabilities 时使用。
// 应与 training_preset_service.py build_training_capabilities() 保持同步。
const FALLBACK_TRAINING_CAPABILITIES = {
  precision_support_by_device: {
    auto: { fp32: true, fp16: true, bf16: false },
    cpu: { fp32: true, fp16: false, bf16: false },
  },
  tiny_support_by_preset: {
    balanced: { available: true, reason_disabled: null },
    high_quality: { available: false, reason_disabled: "当前便携包未提供该套餐的 tiny 底模。" },
    light: { available: false, reason_disabled: "当前便携包未提供该套餐的 tiny 底模。" },
  },
  encoder_asset_support: {
    balanced: { available: true, reason_disabled: null },
    high_quality: { available: true, reason_disabled: null },
    light: { available: true, reason_disabled: null },
  },
  diffusion_asset_support: {
    balanced: { available: true, reason_disabled: null },
    high_quality: { available: true, reason_disabled: null },
    light: { available: false, reason_disabled: "当前便携包未提供该套餐的扩散底模。" },
  },
};

// ── 全局状态 ────────────────────────────────────────────────────────────────
const state = {
  settings: null,
  runtime: null,
  trainingPresets: null,
  trainingCapabilities: null,
  datasets: [],
  models: [],
  selectedDatasetId: null,
  selectedDatasetVersionId: null,
  selectedModelId: null,
  selectedResumeVersionId: null,
  selectedInferenceProfileId: null,
  trainingPresetId: "balanced",
  trainingUseTiny: false,
  trainingF0: "rmvpe",
  inferenceF0: "rmvpe",
  trainingTaskId: null,
  trainingTaskStatus: "idle",
  trainingWs: null,
  trainingPollTimer: null,
  inferenceTaskId: null,
  inferenceTaskStatus: "idle",
  inferenceWs: null,
  inferencePollTimer: null,
  inferenceFile: null,
  inferenceModelLoaded: false,
  inferenceUseDiffusion: false,
  preprocessTaskId: null,
  preprocessTaskStatus: "idle",
  preprocessWs: null,
  preprocessPollTimer: null,
  preprocessSelection: null,
  preprocessResult: null,
  preprocessEngine: "demucs",
  preprocessAutoExtract: false,
  dsExtractFile: null,
  dsExtractDatasetId: null,
  dsExtractTaskId: null,
  dsExtractTaskStatus: "idle",
  dsExtractWs: null,
  dsExtractPollTimer: null,
  dsExtractSelection: null,
  dsExtractResult: null,
  dsExtractEngine: "demucs",
  dsExtractAutoSegment: true,
  dsExtractConfirmPending: false,
  isUnloading: false,
  objectUrls: [],
};

// ── DOM 引用 ────────────────────────────────────────────────────────────────
const els = {
  // Topbar
  runtimeStatusChip:        document.getElementById("runtimeStatusChip"),
  runtimeLibraryText:       document.getElementById("runtimeLibraryText"),
  settingsTrigger:          document.getElementById("settingsTrigger"),

  // Settings Drawer
  settingsOverlay:          document.getElementById("settingsOverlay"),
  settingsDrawer:           document.getElementById("settingsDrawer"),
  settingsCloseBtn:         document.getElementById("settingsCloseBtn"),
  runtimeGpuText:           document.getElementById("runtimeGpuText"),
  runtimeCudaText:          document.getElementById("runtimeCudaText"),
  runtimeCompatibilityText: document.getElementById("runtimeCompatibilityText"),
  dataRootText:             document.getElementById("dataRootText"),
  settingsDataRootInput:    document.getElementById("settingsDataRootInput"),
  settingsDefaultF0Select:  document.getElementById("settingsDefaultF0Select"),
  settingsDefaultStepsInput:        document.getElementById("settingsDefaultStepsInput"),
  settingsCheckpointIntervalInput:  document.getElementById("settingsCheckpointIntervalInput"),
  settingsCheckpointKeepInput:      document.getElementById("settingsCheckpointKeepInput"),
  saveSettingsButton:       document.getElementById("saveSettingsButton"),
  refreshAllButton:         document.getElementById("refreshAllButton"),
  settingsHintText:         document.getElementById("settingsHintText"),
  defaultF0Text:            document.getElementById("defaultF0Text"),
  defaultCheckpointText:    document.getElementById("defaultCheckpointText"),

  // Datasets Tab
  datasetCountChip:         document.getElementById("datasetCountChip"),
  datasetNameInput:         document.getElementById("datasetNameInput"),
  datasetSpeakerInput:      document.getElementById("datasetSpeakerInput"),
  datasetDescriptionInput:  document.getElementById("datasetDescriptionInput"),
  createDatasetButton:      document.getElementById("createDatasetButton"),
  datasetCards:             document.getElementById("datasetCards"),
  datasetDetailArea:        document.getElementById("datasetDetailArea"),
  datasetDetailContent:     document.getElementById("datasetDetailContent"),
  selectedDatasetTitle:     document.getElementById("selectedDatasetTitle"),
  selectedDatasetMeta:      document.getElementById("selectedDatasetMeta"),
  datasetUploadInput:       document.getElementById("datasetUploadInput"),
  datasetAutoSegmentCheckbox: document.getElementById("datasetAutoSegmentCheckbox"),
  uploadDatasetButton:      document.getElementById("uploadDatasetButton"),
  rerunSegmentButton:       document.getElementById("rerunSegmentButton"),
  datasetUploadHintText:    document.getElementById("datasetUploadHintText"),
  dsExtractFileInput:       document.getElementById("dsExtractFileInput"),
  dsExtractAcceptedText:    document.getElementById("dsExtractAcceptedText"),
  dsExtractFileNameText:    document.getElementById("dsExtractFileNameText"),
  dsExtractEngineCards:     document.getElementById("dsExtractEngineCards"),
  dsExtractEngineNote:      document.getElementById("dsExtractEngineNote"),
  dsExtractAutoSegmentCheckbox: document.getElementById("dsExtractAutoSegmentCheckbox"),
  dsExtractStartButton:     document.getElementById("dsExtractStartButton"),
  dsExtractStatusChip:      document.getElementById("dsExtractStatusChip"),
  dsExtractHintText:        document.getElementById("dsExtractHintText"),
  dsExtractProgressBar:     document.getElementById("dsExtractProgressBar"),
  dsExtractProgressText:    document.getElementById("dsExtractProgressText"),
  dsExtractStageText:       document.getElementById("dsExtractStageText"),
  dsExtractLogConsole:      document.getElementById("dsExtractLogConsole"),
  dsExtractWarningText:     document.getElementById("dsExtractWarningText"),
  dsExtractCompareCard:     document.getElementById("dsExtractCompareCard"),
  dsExtractSelectionChip:   document.getElementById("dsExtractSelectionChip"),
  dsExtractOriginalCard:    document.getElementById("dsExtractOriginalCard"),
  dsExtractVocalsCard:      document.getElementById("dsExtractVocalsCard"),
  dsExtractOriginalPlayer:  document.getElementById("dsExtractOriginalPlayer"),
  dsExtractVocalsPlayer:    document.getElementById("dsExtractVocalsPlayer"),
  dsExtractOriginalMeta:    document.getElementById("dsExtractOriginalMeta"),
  dsExtractVocalsMeta:      document.getElementById("dsExtractVocalsMeta"),
  dsExtractUseOriginalButton: document.getElementById("dsExtractUseOriginalButton"),
  dsExtractUseVocalsButton: document.getElementById("dsExtractUseVocalsButton"),
  dsExtractConfirmButton:   document.getElementById("dsExtractConfirmButton"),
  datasetFileCountText:     document.getElementById("datasetFileCountText"),
  datasetFilesList:         document.getElementById("datasetFilesList"),
  datasetSegmentStatsText:  document.getElementById("datasetSegmentStatsText"),
  datasetSegmentsList:      document.getElementById("datasetSegmentsList"),
  datasetVersionLabelInput: document.getElementById("datasetVersionLabelInput"),
  createDatasetVersionButton: document.getElementById("createDatasetVersionButton"),
  datasetVersionsList:      document.getElementById("datasetVersionsList"),

  // Training Tab
  trainingTaskStatusText:   document.getElementById("trainingTaskStatusText"),
  trainingDatasetVersionSelect: document.getElementById("trainingDatasetVersionSelect"),
  trainingModelNameInput:   document.getElementById("trainingModelNameInput"),
  trainingModeSelect:       document.getElementById("trainingModeSelect"),
  resumeFields:             document.getElementById("resumeFields"),
  trainingResumeVersionLabel: document.getElementById("trainingResumeVersionLabel"),
  trainingResumeCheckpointField: document.getElementById("trainingResumeCheckpointField"),
  trainingResumeVersionSelect:   document.getElementById("trainingResumeVersionSelect"),
  trainingResumeCheckpointSelect: document.getElementById("trainingResumeCheckpointSelect"),
  trainingDeviceSelect:     document.getElementById("trainingDeviceSelect"),
  trainingTargetTitle:      document.getElementById("trainingTargetTitle"),
  trainingTargetModeText:   document.getElementById("trainingTargetModeText"),
  trainingTargetVersionText: document.getElementById("trainingTargetVersionText"),
  trainingHistoryHintText:  document.getElementById("trainingHistoryHintText"),
  trainingPresetSection:    document.getElementById("trainingPresetSection"),
  trainingPresetContextChip: document.getElementById("trainingPresetContextChip"),
  trainingPresetCards:      document.getElementById("trainingPresetCards"),
  trainingPresetDerivedText: document.getElementById("trainingPresetDerivedText"),
  trainingPresetSummaryText: document.getElementById("trainingPresetSummaryText"),
  trainingArchitectureReadonly: document.getElementById("trainingArchitectureReadonly"),
  trainingReadonlySourceChip: document.getElementById("trainingReadonlySourceChip"),
  trainingReadonlyPresetText: document.getElementById("trainingReadonlyPresetText"),
  trainingReadonlyEncoderText: document.getElementById("trainingReadonlyEncoderText"),
  trainingReadonlyTinyText: document.getElementById("trainingReadonlyTinyText"),
  trainingReadonlySslText:  document.getElementById("trainingReadonlySslText"),
  trainingReadonlyGinText:  document.getElementById("trainingReadonlyGinText"),
  trainingReadonlyFilterText: document.getElementById("trainingReadonlyFilterText"),
  trainingReadonlySummaryText: document.getElementById("trainingReadonlySummaryText"),
  trainingPredictorCards:   document.getElementById("trainingPredictorCards"),
  trainingPredictorNote:    document.getElementById("trainingPredictorNote"),
  trainingStepsInput:       document.getElementById("trainingStepsInput"),
  trainingCheckpointIntervalInput: document.getElementById("trainingCheckpointIntervalInput"),
  trainingCheckpointKeepInput:     document.getElementById("trainingCheckpointKeepInput"),
  trainingMainBatchSizeInput: document.getElementById("trainingMainBatchSizeInput"),
  trainingMainPrecisionSelect: document.getElementById("trainingMainPrecisionSelect"),
  trainingMainAllInMemCheckbox: document.getElementById("trainingMainAllInMemCheckbox"),
  trainingUseTinyCheckbox:  document.getElementById("trainingUseTinyCheckbox"),
  trainingUseTinyHintText:  document.getElementById("trainingUseTinyHintText"),
  trainingLearningRateInput: document.getElementById("trainingLearningRateInput"),
  trainingLogIntervalInput: document.getElementById("trainingLogIntervalInput"),
  trainingMainRuntimeHintText: document.getElementById("trainingMainRuntimeHintText"),
  trainingDiffusionModeSelect: document.getElementById("trainingDiffusionModeSelect"),
  trainingDiffBatchSizeInput: document.getElementById("trainingDiffBatchSizeInput"),
  trainingDiffAmpDtypeSelect: document.getElementById("trainingDiffAmpDtypeSelect"),
  trainingDiffCacheAllDataCheckbox: document.getElementById("trainingDiffCacheAllDataCheckbox"),
  trainingDiffCacheDeviceSelect: document.getElementById("trainingDiffCacheDeviceSelect"),
  trainingDiffNumWorkersInput: document.getElementById("trainingDiffNumWorkersInput"),
  trainingDiffusionHintText: document.getElementById("trainingDiffusionHintText"),
  startTrainingButton:      document.getElementById("startTrainingButton"),
  trainingHintText:         document.getElementById("trainingHintText"),
  trainingProgressBar:      document.getElementById("trainingProgressBar"),
  trainingProgressText:     document.getElementById("trainingProgressText"),
  trainingStageText:        document.getElementById("trainingStageText"),
  trainingLogConsole:       document.getElementById("trainingLogConsole"),
  trainingResultVersionText:    document.getElementById("trainingResultVersionText"),
  trainingResultCheckpointText: document.getElementById("trainingResultCheckpointText"),
  trainingResultDeviceText:     document.getElementById("trainingResultDeviceText"),
  trainingCompleteCard:     document.getElementById("trainingCompleteCard"),
  trainingCompleteTitle:    document.getElementById("trainingCompleteTitle"),
  trainingCompleteMeta:     document.getElementById("trainingCompleteMeta"),
  jumpToInferenceButton:    document.getElementById("jumpToInferenceButton"),

  // Model Library
  modelCountChip:           document.getElementById("modelCountChip"),
  modelCards:               document.getElementById("modelCards"),
  modelVersionDetail:       document.getElementById("modelVersionDetail"),
  selectedModelTitle:       document.getElementById("selectedModelTitle"),
  selectedModelMeta:        document.getElementById("selectedModelMeta"),
  modelVersionsList:        document.getElementById("modelVersionsList"),

  // Inference Tab
  inferenceTaskStatusText:  document.getElementById("inferenceTaskStatusText"),
  inferenceLoadStatusChip:  document.getElementById("inferenceLoadStatusChip"),
  inferenceProfileSelect:   document.getElementById("inferenceProfileSelect"),
  inferenceSpeakerSelect:   document.getElementById("inferenceSpeakerSelect"),
  inferenceDeviceSelect:    document.getElementById("inferenceDeviceSelect"),
  inferenceDiffusionCard:   document.getElementById("inferenceDiffusionCard"),
  inferenceUseDiffusionCheckbox: document.getElementById("inferenceUseDiffusionCheckbox"),
  inferenceDiffusionHintText: document.getElementById("inferenceDiffusionHintText"),
  loadInferenceModelButton: document.getElementById("loadInferenceModelButton"),
  unloadInferenceModelButton: document.getElementById("unloadInferenceModelButton"),
  inferenceModelHintText:   document.getElementById("inferenceModelHintText"),
  inferencePrepPanel:       document.getElementById("inferencePrepPanel"),
  preprocessStatusChip:     document.getElementById("preprocessStatusChip"),
  inferenceRunPanel:        document.getElementById("inferenceRunPanel"),
  inferenceFileInput:       document.getElementById("inferenceFileInput"),
  inferenceFileNameText:    document.getElementById("inferenceFileNameText"),
  acceptedMediaText:        document.getElementById("acceptedMediaText"),
  preprocessAutoCheckbox:   document.getElementById("preprocessAutoCheckbox"),
  separatorEngineCards:     document.getElementById("separatorEngineCards"),
  separatorEngineNote:      document.getElementById("separatorEngineNote"),
  startPreprocessButton:    document.getElementById("startPreprocessButton"),
  preprocessHintText:       document.getElementById("preprocessHintText"),
  preprocessProgressBar:    document.getElementById("preprocessProgressBar"),
  preprocessProgressText:   document.getElementById("preprocessProgressText"),
  preprocessStageText:      document.getElementById("preprocessStageText"),
  preprocessLogConsole:     document.getElementById("preprocessLogConsole"),
  preprocessWarningText:    document.getElementById("preprocessWarningText"),
  preprocessCompareCard:    document.getElementById("preprocessCompareCard"),
  preprocessSelectionChip:  document.getElementById("preprocessSelectionChip"),
  preprocessOriginalCard:   document.getElementById("preprocessOriginalCard"),
  preprocessVocalsCard:     document.getElementById("preprocessVocalsCard"),
  preprocessOriginalPlayer: document.getElementById("preprocessOriginalPlayer"),
  preprocessVocalsPlayer:   document.getElementById("preprocessVocalsPlayer"),
  preprocessOriginalMeta:   document.getElementById("preprocessOriginalMeta"),
  preprocessVocalsMeta:     document.getElementById("preprocessVocalsMeta"),
  useOriginalButton:        document.getElementById("useOriginalButton"),
  useVocalsButton:          document.getElementById("useVocalsButton"),
  inferenceReadyChip:       document.getElementById("inferenceReadyChip"),
  selectedInputVariantText: document.getElementById("selectedInputVariantText"),
  selectedInputSourceText:  document.getElementById("selectedInputSourceText"),
  inferenceReadyHintText:   document.getElementById("inferenceReadyHintText"),
  inferencePredictorCards:  document.getElementById("inferencePredictorCards"),
  inferencePredictorNote:   document.getElementById("inferencePredictorNote"),
  inferenceTranInput:       document.getElementById("inferenceTranInput"),
  inferenceSliceInput:      document.getElementById("inferenceSliceInput"),
  inferenceNoiseInput:      document.getElementById("inferenceNoiseInput"),
  inferencePadInput:        document.getElementById("inferencePadInput"),
  startInferenceButton:     document.getElementById("startInferenceButton"),
  inferenceProgressBar:     document.getElementById("inferenceProgressBar"),
  inferenceProgressText:    document.getElementById("inferenceProgressText"),
  inferenceStageText:       document.getElementById("inferenceStageText"),
  inferenceLogConsole:      document.getElementById("inferenceLogConsole"),
  inferenceResultPlayer:    document.getElementById("inferenceResultPlayer"),
  inferenceDownloadLink:    document.getElementById("inferenceDownloadLink"),
};

// ── 工具函数 ─────────────────────────────────────────────────────────────────
function setText(el, value) {
  if (el) el.textContent = value ?? "--";
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "--";
  const units = ["B", "KB", "MB", "GB"];
  let v = bytes, i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return "--";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  return `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(0)}s`;
}

function appendLog(el, message) {
  if (!message || !el) return;
  const current = el.textContent.trim();
  const lines = current ? current.split("\n") : [];
  if (lines[lines.length - 1] === message) return;
  el.textContent = (current && !current.startsWith("等待")) ? `${current}\n${message}` : message;
  el.scrollTop = el.scrollHeight;
}

async function requestJSON(url, options = {}) {
  const resp = await fetch(url, options);
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.detail || data.message || `请求失败：HTTP ${resp.status}`);
  return data;
}

function populateSelect(selectEl, options, preferredValue = "", fallbackLabel = "暂无可选项") {
  if (!selectEl) return;
  selectEl.innerHTML = "";
  if (!Array.isArray(options) || !options.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = fallbackLabel;
    selectEl.appendChild(opt);
    selectEl.disabled = true;
    return;
  }
  selectEl.disabled = false;
  options.forEach((item) => {
    const opt = document.createElement("option");
    opt.value = item.value ?? item.id ?? item;
    opt.textContent = item.label ?? item.name ?? item.value ?? item;
    selectEl.appendChild(opt);
  });
  selectEl.value = preferredValue || selectEl.options[0].value;
}

function populateSelectWithDisabled(selectEl, options, preferredValue = "", fallbackLabel = "暂无可选项") {
  if (!selectEl) return;
  selectEl.innerHTML = "";
  if (!Array.isArray(options) || !options.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = fallbackLabel;
    selectEl.appendChild(opt);
    selectEl.disabled = true;
    return;
  }
  selectEl.disabled = false;
  let selectedValue = preferredValue;
  options.forEach((item, index) => {
    const opt = document.createElement("option");
    opt.value = item.value ?? item.id ?? item;
    opt.textContent = item.label ?? item.name ?? item.value ?? item;
    if (item.disabled) opt.disabled = true;
    selectEl.appendChild(opt);
    if ((!selectedValue || item.disabled) && index === 0) {
      selectedValue = opt.value;
    }
  });
  if (selectedValue && Array.from(selectEl.options).some(option => option.value === selectedValue && !option.disabled)) {
    selectEl.value = selectedValue;
    return;
  }
  const firstEnabled = Array.from(selectEl.options).find(option => !option.disabled);
  selectEl.value = firstEnabled?.value || selectEl.options[0]?.value || "";
}

function datasetById(id)  { return state.datasets.find(d => d.id === id) || null; }
function modelById(id)    { return state.models.find(m => m.id === id) || null; }
function versionById(vid) {
  for (const m of state.models) {
    for (const v of (m.versions || [])) {
      if (v.id === vid) return { model: m, version: v };
    }
  }
  return null;
}

function predictorOptions() {
  return Object.keys(PREDICTOR_META).map(k => ({ value: k, label: PREDICTOR_META[k].title }));
}

function trainingPresets() {
  const presets = state.trainingPresets;
  return Array.isArray(presets) && presets.length ? presets : FALLBACK_TRAINING_PRESETS;
}

function trainingCapabilities() {
  return state.trainingCapabilities || FALLBACK_TRAINING_CAPABILITIES;
}

function getTrainingPresetById(presetId) {
  return trainingPresets().find(item => item.id === presetId) || null;
}

// 从版本记录推断套餐 ID。优先读 main_preset_id，
// 对缺少该字段的历史版本按 speech_encoder 兜底推断。
function inferPresetIdFromVersion(version) {
  if (!version) return state.trainingPresetId || "balanced";
  if (version.main_preset_id) return version.main_preset_id;
  if (version.speech_encoder === "whisper-ppg") return "high_quality";
  if (version.speech_encoder === "hubertsoft") return "light";
  return "balanced";
}

function currentTrainingMode() {
  return els.trainingModeSelect?.value || "new";
}

function trainingSourceVersionDetail() {
  return versionById(state.selectedResumeVersionId)?.version || null;
}

function profileHasDiffusion(profile) {
  if (!profile) return false;
  return profile.diffusion_status === "trained" && Boolean(profile.diffusion_model_path);
}

function selectedInferenceProfile() {
  return (state.runtime?.profiles || []).find(profile => profile.id === (els.inferenceProfileSelect?.value || state.selectedInferenceProfileId)) || null;
}

function presetMeta(presetId) {
  return TRAINING_PRESET_META[presetId] || TRAINING_PRESET_META.balanced;
}

function tinySupportForPreset(presetId) {
  const tinySupport = trainingCapabilities().tiny_support_by_preset || {};
  return tinySupport[presetId] || { available: presetId === "balanced", reason_disabled: presetId === "balanced" ? null : "当前便携包未提供该套餐的 tiny 底模。" };
}

function precisionOptionsForDevice(deviceValue) {
  const capabilityMap = trainingCapabilities().precision_support_by_device || {};
  const normalizedDevice = deviceValue || "auto";
  const fallbackAuto = {
    fp32: true,
    fp16: normalizedDevice !== "cpu" && Boolean(state.runtime?.gpu_compat?.cuda_available || state.runtime?.gpu_compatibility?.cuda_available),
    bf16: false,
  };
  const support = capabilityMap[normalizedDevice] || capabilityMap.auto || fallbackAuto;
  return [
    { value: "fp32", label: "fp32", disabled: support.fp32 === false },
    { value: "fp16", label: "fp16", disabled: support.fp16 === false },
    { value: "bf16", label: "bf16", disabled: support.bf16 === false },
  ];
}

function diffusionStatusLabel(status) {
  if (status === "trained") return "扩散已训练";
  if (status === "training") return "扩散训练中";
  if (status === "failed") return "扩散训练失败";
  return "扩散未训练";
}

function trainingModeLabel(mode) {
  if (mode === "resume") return "继续训练";
  if (mode === "diffusion_only") return "仅训练扩散";
  return "完全重训";
}

/** 是否是活跃中的任务状态 */
function isTaskActive(status) {
  return !["completed", "failed", "idle", ""].includes(status || "idle");
}

function hasActiveGpuTask() {
  return [
    state.trainingTaskStatus,
    state.inferenceTaskStatus,
    state.preprocessTaskStatus,
    state.dsExtractTaskStatus,
  ].some(isTaskActive);
}

/** 释放 createObjectURL 产生的 URL */
function clearObjectUrls() {
  state.objectUrls.forEach(u => { try { URL.revokeObjectURL(u); } catch (_) {} });
  state.objectUrls = [];
}

function clearMediaElementSource(el) {
  if (!el) return;
  el.pause?.();
  el.removeAttribute("src");
  el.load?.();
}

function engineLabel(engine) {
  return SEPARATOR_META[engine]?.title || engine || "--";
}

function separatorOptionByKey(key) {
  return (state.runtime?.separator_engines || []).find(option => option.value === key) || null;
}

function selectedVariantLabel(variant) {
  if (variant === "vocals") return "提取人声";
  if (variant === "original") return "原始音频";
  return "--";
}

function canAdvanceWithoutPreprocess() {
  return Boolean(state.inferenceFile) && !state.preprocessAutoExtract;
}

function isPreprocessReadySelection() {
  return ["original", "vocals"].includes(state.preprocessSelection || "");
}

function canRunInference() {
  return Boolean(state.inferenceFile) && state.inferenceModelLoaded && isPreprocessReadySelection();
}

// ── Pipeline 导航 ─────────────────────────────────────────────────────────────
function setActiveView(tabName) {
  document.querySelectorAll(".pipeline-card[data-tab]").forEach(card => {
    card.classList.toggle("is-active", card.dataset.tab === tabName);
  });
  document.querySelectorAll(".tab-panel").forEach(panel => {
    panel.classList.toggle("is-active", panel.dataset.panel === tabName);
  });
  updatePipelineState();
}

function updatePipelineState() {
  const pStepDs = document.getElementById("pipelineStepDataset");
  const pStepTr = document.getElementById("pipelineStepTraining");
  const pStepInf = document.getElementById("pipelineStepInference");
  const pStatusDs = document.getElementById("pipelineStatusDataset");
  const pStatusTr = document.getElementById("pipelineStatusTraining");
  const pStatusInf = document.getElementById("pipelineStatusInference");
  const cardDs = document.querySelector('.pipeline-card[data-tab="datasets"]');
  const cardTr = document.querySelector('.pipeline-card[data-tab="training"]');
  const cardInf = document.querySelector('.pipeline-card[data-tab="inference"]');

  // Dataset stage
  const dsCount = state.datasets.length;
  const hasVersions = state.datasets.some(ds => (ds.versions || []).length > 0);
  const isExtracting = isTaskActive(state.dsExtractTaskStatus);
  if (pStatusDs) {
    if (isExtracting) pStatusDs.textContent = "素材提取中…";
    else pStatusDs.textContent = dsCount ? `${dsCount} 个数据集` + (hasVersions ? " · 已有版本" : "") : "准备数据";
  }
  if (cardDs) cardDs.classList.toggle("is-complete", hasVersions);
  if (pStepDs && hasVersions) pStepDs.textContent = "✓";
  else if (pStepDs) pStepDs.textContent = "01";

  // Training stage
  const modelCount = state.models.length;
  const isTraining = isTaskActive(state.trainingTaskStatus);
  if (pStatusTr) {
    if (isTraining) pStatusTr.textContent = "训练中…";
    else if (modelCount) pStatusTr.textContent = `${modelCount} 个模型`;
    else pStatusTr.textContent = "配置训练";
  }
  if (cardTr) cardTr.classList.toggle("is-complete", modelCount > 0 && !isTraining);
  if (pStepTr && modelCount > 0 && !isTraining) pStepTr.textContent = "✓";
  else if (pStepTr) pStepTr.textContent = "02";

  // Inference stage
  const isInferring = isTaskActive(state.inferenceTaskStatus);
  if (pStatusInf) {
    if (isInferring) pStatusInf.textContent = "推理中…";
    else if (state.inferenceModelLoaded) pStatusInf.textContent = "模型已加载";
    else pStatusInf.textContent = "生成结果";
  }
  if (cardInf) cardInf.classList.toggle("is-complete", false);
  if (pStepInf) pStepInf.textContent = "03";
}

function bindTabNav() {
  document.querySelectorAll(".pipeline-card[data-tab]").forEach(card => {
    card.addEventListener("click", () => setActiveView(card.dataset.tab));
  });
}

// ── 设置抽屉 ─────────────────────────────────────────────────────────────────
function openSettings() {
  els.settingsDrawer.classList.add("is-open");
  els.settingsOverlay.classList.add("is-open");
}

function closeSettings() {
  els.settingsDrawer.classList.remove("is-open");
  els.settingsOverlay.classList.remove("is-open");
}

// ── F0 Predictor 卡片渲染 ────────────────────────────────────────────────────
function renderPredictorCards(containerEl, noteEl, currentValue, onSelect) {
  if (!containerEl) return;
  containerEl.innerHTML = "";

  const primaryCards = document.createElement("div");
  primaryCards.className = "predictor-cards-primary";
  primaryCards.style.display = "flex";
  primaryCards.style.flexDirection = "column";
  primaryCards.style.gap = "8px";

  const moreGroup = document.createElement("div");
  moreGroup.className = "predictor-more-group hidden";
  moreGroup.style.display = "flex";
  moreGroup.style.flexDirection = "column";
  moreGroup.style.gap = "6px";

  const moreToggle = document.createElement("button");
  moreToggle.className = "predictor-more-toggle";
  moreToggle.type = "button";
  moreToggle.textContent = "显示更多选项（Harvest · DIO · PM）";

  let moreShown = false;
  moreToggle.addEventListener("click", () => {
    moreShown = !moreShown;
    moreGroup.classList.toggle("hidden", !moreShown);
    moreToggle.textContent = moreShown
      ? "收起"
      : "显示更多选项（Harvest · DIO · PM）";
  });

  function makeCard(key, meta) {
    const label = document.createElement("label");
    label.className = `predictor-card${key === currentValue ? " is-selected" : ""}${meta.tier === "recommended" ? " is-recommended" : ""}`;

    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = `f0_${containerEl.id}`;
    radio.value = key;
    radio.checked = key === currentValue;

    radio.addEventListener("change", () => {
      containerEl.querySelectorAll(".predictor-card").forEach(c => c.classList.remove("is-selected"));
      label.classList.add("is-selected");
      onSelect(key);
      if (noteEl) {
        noteEl.innerHTML = `<strong>${meta.title}</strong> — ${meta.description}`;
      }
    });

    const body = document.createElement("div");
    body.className = "predictor-card-body";

    const nameLine = document.createElement("div");
    nameLine.className = "predictor-name";
    nameLine.textContent = meta.title;

    if (meta.badge) {
      const badge = document.createElement("span");
      badge.className = "predictor-badge";
      badge.textContent = meta.badge;
      nameLine.appendChild(badge);
    }

    const desc = document.createElement("div");
    desc.className = "predictor-desc";
    desc.textContent = meta.description;

    body.appendChild(nameLine);
    body.appendChild(desc);
    label.appendChild(radio);
    label.appendChild(body);

    return label;
  }

  Object.entries(PREDICTOR_META).forEach(([key, meta]) => {
    const card = makeCard(key, meta);
    if (PREDICTOR_PRIMARY.includes(key)) {
      primaryCards.appendChild(card);
    } else {
      moreGroup.appendChild(card);
    }
  });

  containerEl.appendChild(primaryCards);
  containerEl.appendChild(moreToggle);
  containerEl.appendChild(moreGroup);

  // 初始化 note
  if (noteEl) {
    const cur = PREDICTOR_META[currentValue];
    if (cur) noteEl.innerHTML = `<strong>${cur.title}</strong> — ${cur.description}`;
  }
}

function renderSeparatorCards(containerEl, noteEl, currentValue, onSelect) {
  if (!containerEl) return;
  containerEl.innerHTML = "";

  const primaryCards = document.createElement("div");
  primaryCards.className = "predictor-cards-primary";
  primaryCards.style.display = "flex";
  primaryCards.style.flexDirection = "column";
  primaryCards.style.gap = "8px";

  const moreGroup = document.createElement("div");
  moreGroup.className = "predictor-more-group hidden";
  moreGroup.style.display = "flex";
  moreGroup.style.flexDirection = "column";
  moreGroup.style.gap = "6px";

  const moreToggle = document.createElement("button");
  moreToggle.className = "predictor-more-toggle";
  moreToggle.type = "button";
  moreToggle.textContent = "显示更多选项（MDX-Net）";

  let moreShown = false;
  moreToggle.addEventListener("click", () => {
    moreShown = !moreShown;
    moreGroup.classList.toggle("hidden", !moreShown);
    moreToggle.textContent = moreShown ? "收起" : "显示更多选项（MDX-Net）";
  });

  function makeCard(key, meta) {
    const runtimeOption = separatorOptionByKey(key);
    const available = runtimeOption?.available !== false;
    const disabledReason = runtimeOption?.reason_disabled || "";
    const label = document.createElement("label");
    label.className = `predictor-card${key === currentValue ? " is-selected" : ""}${meta.tier === "recommended" ? " is-recommended" : ""}${available ? "" : " is-disabled"}`;

    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = `separator_${containerEl.id}`;
    radio.value = key;
    radio.checked = key === currentValue;
    radio.disabled = !available;

    radio.addEventListener("change", () => {
      if (!available) return;
      containerEl.querySelectorAll(".predictor-card").forEach(card => card.classList.remove("is-selected"));
      label.classList.add("is-selected");
      onSelect(key);
      if (noteEl) {
        noteEl.innerHTML = `<strong>${meta.title}</strong> — ${meta.description}`;
      }
    });

    const body = document.createElement("div");
    body.className = "predictor-card-body";

    const nameLine = document.createElement("div");
    nameLine.className = "predictor-name";
    nameLine.textContent = meta.title;

    if (meta.badge) {
      const badge = document.createElement("span");
      badge.className = "predictor-badge";
      badge.textContent = meta.badge;
      nameLine.appendChild(badge);
    }

    const desc = document.createElement("div");
    desc.className = "predictor-desc";
    desc.textContent = available ? meta.description : `${meta.description} ${disabledReason}`.trim();

    body.appendChild(nameLine);
    body.appendChild(desc);
    label.appendChild(radio);
    label.appendChild(body);
    return label;
  }

  Object.entries(SEPARATOR_META).forEach(([key, meta]) => {
    const card = makeCard(key, meta);
    if (key === "demucs") {
      primaryCards.appendChild(card);
    } else {
      moreGroup.appendChild(card);
    }
  });

  containerEl.appendChild(primaryCards);
  containerEl.appendChild(moreToggle);
  containerEl.appendChild(moreGroup);

  if (noteEl) {
    const cur = SEPARATOR_META[currentValue];
    const runtimeOption = separatorOptionByKey(currentValue);
    if (cur) noteEl.innerHTML = `<strong>${cur.title}</strong> — ${cur.description}${runtimeOption?.available === false ? ` 当前不可用：${runtimeOption.reason_disabled}` : ""}`;
  }
}

// ── 渲染函数 ─────────────────────────────────────────────────────────────────
function applyOverview(payload) {
  state.settings = payload.settings;
  state.runtime  = payload.runtime;
  state.trainingPresets = payload.training_presets;
  state.trainingCapabilities = payload.training_capabilities;
  state.datasets = payload.datasets || [];
  state.models   = payload.models || [];

  // Sync model loaded state from backend
  if (payload.runtime?.current_profile) {
    const loadedId = payload.runtime.current_profile.id;
    state.inferenceModelLoaded = Boolean(loadedId && loadedId === state.selectedInferenceProfileId);
  } else {
    state.inferenceModelLoaded = false;
  }

  if (!datasetById(state.selectedDatasetId))
    state.selectedDatasetId = state.datasets[0]?.id || null;

  const ds = datasetById(state.selectedDatasetId);
  if (!(ds?.versions || []).find(v => v.id === state.selectedDatasetVersionId))
    state.selectedDatasetVersionId = ds?.versions?.[0]?.id || null;

  if (!modelById(state.selectedModelId))
    state.selectedModelId = state.models[0]?.id || null;

  renderRuntime();
  renderSettings();
  renderDatasets();
  renderDatasetDetail();
  renderTrainingConsole();
  renderModels();
  renderModelDetail();
  renderInferenceWorkbench();
  toggleBusyActions();
  updatePipelineState();
}

async function refreshOverview() {
  applyOverview(await requestJSON("/api/library/overview"));
}

function renderRuntime() {
  const compat = state.runtime?.gpu_compat || state.runtime?.gpu_compatibility || {};
  const ok = compat.compatible;

  // Topbar chip
  const dot = els.runtimeStatusChip?.querySelector(".chip-dot");
  if (dot) {
    dot.className = `chip-dot ${ok ? "ok" : "warn"}`;
  }
  if (els.runtimeStatusChip) {
    const txt = els.runtimeStatusChip.childNodes;
    // Replace text node (last child or only text)
    const textNode = Array.from(els.runtimeStatusChip.childNodes).find(n => n.nodeType === 3);
    if (textNode) textNode.textContent = ok ? " GPU Ready" : " Needs Attention";
  }

  setText(els.runtimeLibraryText, `${state.datasets.length} datasets · ${state.models.length} models`);

  // Drawer details
  setText(els.runtimeGpuText, compat.gpu_name || "未检测到 GPU");
  setText(els.runtimeCudaText, `Torch CUDA ${compat.torch_cuda_version || "--"}`);
  setText(els.runtimeCompatibilityText, compat.message || "等待运行时回报。");
  setText(els.dataRootText, state.runtime?.data_root || "--");
}

function renderSettings() {
  const s = state.settings || {};
  if (els.settingsDataRootInput) els.settingsDataRootInput.value = s.app?.data_root || "workspace_data";
  populateSelect(els.settingsDefaultF0Select, predictorOptions(), s.training_defaults?.f0_predictor || "rmvpe");
  if (els.settingsDefaultStepsInput) els.settingsDefaultStepsInput.value = s.training_defaults?.step_count || 2000;
  if (els.settingsCheckpointIntervalInput) els.settingsCheckpointIntervalInput.value = s.training_defaults?.checkpoint_interval_steps || 500;
  if (els.settingsCheckpointKeepInput) els.settingsCheckpointKeepInput.value = s.training_defaults?.checkpoint_keep_last || 5;
  setText(els.defaultF0Text, s.training_defaults?.f0_predictor || "rmvpe");
  setText(els.defaultCheckpointText, `每 ${s.training_defaults?.checkpoint_interval_steps || 500} 步 / 保留 ${s.training_defaults?.checkpoint_keep_last || 5} 个`);
}

function renderDatasets() {
  setText(els.datasetCountChip, String(state.datasets.length));

  if (!state.datasets.length) {
    els.datasetCards.innerHTML = `
      <div class="empty-state">
        <span class="empty-icon">🎙</span>
        <p class="empty-title">还没有数据集</p>
        <p class="hint-text">填写上方表单，创建第一个数据集。</p>
      </div>`;
    return;
  }

  els.datasetCards.innerHTML = state.datasets.map(ds => `
    <button class="dataset-card ${ds.id === state.selectedDatasetId ? "is-active" : ""}" type="button" data-dataset-id="${ds.id}">
      <div class="card-topline">
        <span class="card-title">${ds.name}</span>
        <span class="panel-kicker">${ds.speaker}</span>
      </div>
      <div class="card-meta">
        <div>${ds.file_count} 个文件 · ${ds.enabled_segment_count || 0}/${ds.segment_count || 0} 片段已启用</div>
        <div>${ds.version_count || 0} 个版本</div>
      </div>
    </button>
  `).join("");
}

function renderDatasetDetail() {
  const ds = datasetById(state.selectedDatasetId);

  // 切换占位 / 详情
  const placeholder = els.datasetDetailArea?.querySelector(".detail-placeholder");
  if (placeholder) placeholder.style.display = ds ? "none" : "flex";
  if (els.datasetDetailContent) {
    els.datasetDetailContent.classList.toggle("hidden", !ds);
  }
  if (!ds) return;

  setText(els.selectedDatasetTitle, ds.name);
  setText(els.selectedDatasetMeta, `${ds.speaker} · ${ds.file_count} 个文件`);
  setText(els.datasetFileCountText, `${(ds.files || []).length} files`);
  setText(els.datasetSegmentStatsText, `${ds.enabled_segment_count || 0} / ${ds.segment_count || 0}`);

  // 原始音频
  els.datasetFilesList.innerHTML = (ds.files || []).length
    ? (ds.files || []).map(f => `
        <article class="file-item">
          <div class="card-topline">
            <span class="card-title">${f.original_name}</span>
            <span>${formatDuration(Number(f.duration_seconds))}</span>
          </div>
          <div class="file-meta">${formatBytes(Number(f.size_bytes))} · ${f.sample_rate} Hz</div>
          <audio class="audio-player" controls preload="metadata" src="/api/datasets/${ds.id}/files/${f.id}/audio"></audio>
        </article>
      `).join("")
    : '<p class="hint-text">这个数据集还没有音频。</p>';

  // 候选片段
  const segs = ds.segments || [];
  const preview = segs.slice(0, 60);
  els.datasetSegmentsList.innerHTML = segs.length
    ? `${segs.length > 60 ? `<p class="hint-text">只展示前 60 条，共 ${segs.length} 条。</p>` : ""}
       ${preview.map(seg => `
         <article class="segment-item">
           <div class="segment-row">
             <strong>${seg.display_name}</strong>
             <label class="checkline">
               <input type="checkbox" data-segment-toggle="${seg.id}" ${seg.enabled ? "checked" : ""}>
               <span>${seg.enabled ? "已启用" : "已禁用"}</span>
             </label>
           </div>
           <div class="segment-meta">${formatDuration(Number(seg.duration_seconds))} · ${seg.energy_db} dB · ${seg.start_ms}ms – ${seg.end_ms}ms</div>
           <audio class="audio-player" controls preload="metadata" src="/api/segments/${seg.id}/audio"></audio>
         </article>
       `).join("")}`
    : '<p class="hint-text">还没有候选片段。上传音频后自动分段，或点击"重新分段"。</p>';

  // 版本列表
  els.datasetVersionsList.innerHTML = (ds.versions || []).length
    ? (ds.versions || []).map(v => `
        <article class="version-item ${v.id === state.selectedDatasetVersionId ? "is-active" : ""}">
          <div class="card-topline">
            <span class="card-title">${v.label}</span>
            <span>${v.segment_count} 个片段</span>
          </div>
          <div class="version-meta">${formatDuration(Number(v.total_duration))} · ${v.created_at || ""}</div>
          <div class="actions compact-actions">
            <button class="ghost-button" type="button" data-use-version="${v.id}">用于训练 →</button>
          </div>
        </article>
      `).join("")
    : '<p class="hint-text">审核完片段后，在上方填写标签，点击"固化为版本"，才能用于训练。</p>';

  renderDatasetExtractPanel();
}

function renderDatasetExtractPanel() {
  const ds = datasetById(state.selectedDatasetId);
  if (!ds) return;

  if (state.dsExtractDatasetId && state.dsExtractDatasetId !== ds.id) {
    resetDatasetExtractUi();
  }

  const availableSeparatorOptions = (state.runtime?.separator_engines || []).filter(option => option.available !== false);
  if (availableSeparatorOptions.length && !availableSeparatorOptions.find(option => option.value === state.dsExtractEngine)) {
    state.dsExtractEngine = availableSeparatorOptions[0].value;
  }

  renderSeparatorCards(
    els.dsExtractEngineCards,
    els.dsExtractEngineNote,
    state.dsExtractEngine,
    (key) => { state.dsExtractEngine = key; }
  );

  if (els.dsExtractAcceptedText) {
    setText(
      els.dsExtractAcceptedText,
      state.runtime?.accepted_media_types?.join(" / ") || "支持 WAV / MP3 / MP4 / MKV / MOV / AVI"
    );
  }
  if (els.dsExtractAutoSegmentCheckbox) {
    els.dsExtractAutoSegmentCheckbox.checked = state.dsExtractAutoSegment;
  }
  if (els.dsExtractFileNameText) {
    if (state.dsExtractFile) {
      setText(els.dsExtractFileNameText, `${state.dsExtractFile.name} · ${formatBytes(state.dsExtractFile.size)}`);
    } else if (state.dsExtractResult?.summary?.source_file) {
      setText(els.dsExtractFileNameText, state.dsExtractResult.summary.source_file);
    } else {
      setText(els.dsExtractFileNameText, "尚未选择素材");
    }
  }

  syncDatasetExtractSummary();
  updateDatasetExtractActions();
}

function closeDatasetExtractRealtime() {
  if (state.dsExtractWs) { state.dsExtractWs.close(); state.dsExtractWs = null; }
  if (state.dsExtractPollTimer) { clearInterval(state.dsExtractPollTimer); state.dsExtractPollTimer = null; }
}

function setDsExtractSelection(variant) {
  state.dsExtractSelection = variant;
  els.dsExtractUseOriginalButton?.classList.toggle("is-active", variant === "original");
  els.dsExtractUseVocalsButton?.classList.toggle("is-active", variant === "vocals");
  els.dsExtractOriginalCard?.classList.toggle("is-selected", variant === "original");
  els.dsExtractVocalsCard?.classList.toggle("is-selected", variant === "vocals");
  if (els.dsExtractSelectionChip) {
    setText(
      els.dsExtractSelectionChip,
      variant === "vocals" ? "将加入数据集" : variant === "original" ? "使用原始音轨" : "未确认"
    );
    els.dsExtractSelectionChip.classList.toggle("is-selected", Boolean(variant));
  }
  syncDatasetExtractSummary();
  updateDatasetExtractActions();
}

function syncDatasetExtractSummary() {
  const hasFile = Boolean(state.dsExtractFile);
  let status = "等待素材";
  let statusClass = "is-warn";
  let hint = "支持录屏视频或原始音频，提取完成后可一键加入当前数据集。";

  if (state.dsExtractConfirmPending) {
    status = "添加中…";
    hint = "正在写入数据集并更新片段，请稍候。";
  } else if (["queued", "running"].includes(state.dsExtractTaskStatus)) {
    status = "提取中…";
    hint = "正在提取音轨并分离人声，完成后会出现试听与确认区域。";
  } else if (state.dsExtractTaskStatus === "completed") {
    status = state.dsExtractSelection ? "已选版本" : "等待确认";
    statusClass = state.dsExtractSelection ? "is-ready" : "is-warn";
    hint = state.dsExtractSelection
      ? `已选择${selectedVariantLabel(state.dsExtractSelection)}，可以添加到当前数据集。`
      : "请先试听并选择要加入数据集的版本。";
  } else if (state.dsExtractTaskStatus === "failed") {
    status = state.dsExtractSelection ? "回退原始音轨" : "仅原始音轨";
    hint = state.dsExtractSelection
      ? "将使用原始音轨加入数据集。"
      : "未能得到可用人声，请改用原始音轨。";
  } else if (hasFile) {
    status = "等待提取";
    hint = "素材已选中，点击“开始提取”即可生成可试听版本。";
  }

  setText(els.dsExtractStatusChip, status);
  els.dsExtractStatusChip?.classList.toggle("is-ready", statusClass === "is-ready");
  els.dsExtractStatusChip?.classList.toggle("is-warn", statusClass === "is-warn");
  setText(els.dsExtractHintText, hint);
}

function resetDatasetExtractUi({ preserveFile = false } = {}) {
  closeDatasetExtractRealtime();
  state.dsExtractTaskId = null;
  state.dsExtractTaskStatus = "idle";
  state.dsExtractSelection = null;
  state.dsExtractResult = null;
  state.dsExtractConfirmPending = false;

  if (!preserveFile) {
    state.dsExtractFile = null;
    state.dsExtractDatasetId = null;
    if (els.dsExtractFileInput) {
      els.dsExtractFileInput.value = "";
    }
  }

  if (els.dsExtractProgressBar) els.dsExtractProgressBar.style.width = "0%";
  setText(els.dsExtractProgressText, "0%");
  setText(els.dsExtractStageText, preserveFile && state.dsExtractFile ? "等待开始提取" : "等待选择素材");
  if (els.dsExtractLogConsole) {
    els.dsExtractLogConsole.textContent = preserveFile && state.dsExtractFile ? "等待点击“开始提取”…" : "等待选择素材…";
  }
  if (els.dsExtractFileNameText) {
    if (state.dsExtractFile) {
      setText(els.dsExtractFileNameText, `${state.dsExtractFile.name} · ${formatBytes(state.dsExtractFile.size)}`);
    } else {
      setText(els.dsExtractFileNameText, "尚未选择素材");
    }
  }
  setText(els.dsExtractWarningText, "");
  els.dsExtractWarningText?.classList.add("hidden");
  els.dsExtractCompareCard?.classList.add("hidden");
  els.dsExtractVocalsCard?.classList.remove("hidden");
  setText(els.dsExtractOriginalMeta, "等待结果");
  setText(els.dsExtractVocalsMeta, "推荐：含伴奏时优先试听");
  clearMediaElementSource(els.dsExtractOriginalPlayer);
  clearMediaElementSource(els.dsExtractVocalsPlayer);
  setDsExtractSelection(null);
  syncDatasetExtractSummary();
  updateDatasetExtractActions();
}

function updateDatasetExtractComparisonUi() {
  const completed = state.dsExtractTaskStatus === "completed";
  const failed = state.dsExtractTaskStatus === "failed";
  const hasResult = completed || failed;
  els.dsExtractCompareCard?.classList.toggle("hidden", !hasResult);
  els.dsExtractVocalsCard?.classList.toggle("hidden", failed);
  if (!hasResult) {
    clearMediaElementSource(els.dsExtractOriginalPlayer);
    clearMediaElementSource(els.dsExtractVocalsPlayer);
    return;
  }

  if (state.dsExtractResult?.original_url) {
    els.dsExtractOriginalPlayer.src = state.dsExtractResult.original_url;
    els.dsExtractOriginalPlayer.load();
  } else {
    clearMediaElementSource(els.dsExtractOriginalPlayer);
  }

  if (completed && state.dsExtractResult?.vocals_url) {
    els.dsExtractVocalsPlayer.src = state.dsExtractResult.vocals_url;
    els.dsExtractVocalsPlayer.load();
  } else {
    clearMediaElementSource(els.dsExtractVocalsPlayer);
  }
}

function updateDatasetExtractActions() {
  const hasFile = Boolean(state.dsExtractFile);
  const busy = hasActiveGpuTask();
  const running = ["queued", "running"].includes(state.dsExtractTaskStatus);
  const canChooseOriginal = state.dsExtractTaskStatus === "completed" || state.dsExtractTaskStatus === "failed";
  const canChooseVocals = state.dsExtractTaskStatus === "completed";
  const canConfirm = Boolean(state.dsExtractSelection) && canChooseOriginal && !running && !busy && !state.dsExtractConfirmPending;

  if (els.dsExtractFileInput) {
    els.dsExtractFileInput.disabled = running || state.dsExtractConfirmPending;
  }
  if (els.dsExtractAutoSegmentCheckbox) {
    els.dsExtractAutoSegmentCheckbox.disabled = running || state.dsExtractConfirmPending;
  }
  if (els.dsExtractStartButton) {
    els.dsExtractStartButton.disabled = !hasFile || busy || state.dsExtractConfirmPending;
    els.dsExtractStartButton.textContent = running ? "提取中…" : "开始提取";
  }
  if (els.dsExtractUseOriginalButton) {
    els.dsExtractUseOriginalButton.disabled = !canChooseOriginal || running || busy || state.dsExtractConfirmPending;
  }
  if (els.dsExtractUseVocalsButton) {
    els.dsExtractUseVocalsButton.disabled = !canChooseVocals || running || busy || state.dsExtractConfirmPending;
  }
  if (els.dsExtractConfirmButton) {
    els.dsExtractConfirmButton.disabled = !canConfirm;
    els.dsExtractConfirmButton.textContent = state.dsExtractConfirmPending ? "添加中…" : "添加到数据集";
  }
}

function updateDsExtractTaskUi(payload) {
  const summary = payload.summary || {};
  const activeEngine = summary.separator_engine || payload.separator_engine || state.dsExtractEngine;
  const progress = Math.max(0, Math.min(100, Number(payload.progress ?? 0)));
  state.dsExtractTaskStatus = payload.status || state.dsExtractTaskStatus;
  state.dsExtractResult = payload;

  if (payload.summary?.source_file && !state.dsExtractFile && els.dsExtractFileNameText) {
    setText(els.dsExtractFileNameText, payload.summary.source_file);
  }
  setText(els.dsExtractStageText, `${payload.stage || payload.status || "--"} · ${payload.current_segment ?? "--"} / ${payload.total_segments ?? "--"}`);
  if (els.dsExtractProgressBar) els.dsExtractProgressBar.style.width = `${progress}%`;
  setText(els.dsExtractProgressText, `${progress.toFixed(0)}%`);
  if (payload.message) appendLog(els.dsExtractLogConsole, payload.message);

  if (["queued", "running"].includes(payload.status)) {
    clearMediaElementSource(els.dsExtractOriginalPlayer);
    clearMediaElementSource(els.dsExtractVocalsPlayer);
    els.dsExtractCompareCard?.classList.add("hidden");
    setDsExtractSelection(null);
    els.dsExtractWarningText?.classList.add("hidden");
    setText(els.dsExtractWarningText, "");
  }

  if (payload.status === "completed") {
    setText(els.dsExtractOriginalMeta, payload.original_label || "原始音轨");
    setText(els.dsExtractVocalsMeta, payload.vocals_label || `来自 ${engineLabel(activeEngine)} 的提取结果`);
    updateDatasetExtractComparisonUi();
  }

  if (payload.status === "failed") {
    setText(els.dsExtractOriginalMeta, payload.original_label || "原始音轨");
    setText(els.dsExtractWarningText, payload.warning || "未检测到清晰人声，建议使用原始音轨。");
    els.dsExtractWarningText?.classList.remove("hidden");
    updateDatasetExtractComparisonUi();
    setDsExtractSelection("original");
  }

  syncDatasetExtractSummary();
  updateDatasetExtractActions();
  toggleBusyActions();
  updatePipelineState();
}

function pollDsExtractTask(taskId) {
  return requestJSON(`/api/preprocess/tasks/${encodeURIComponent(taskId)}`)
    .then(payload => {
      updateDsExtractTaskUi(payload);
      if (["completed", "failed"].includes(payload.status)) closeDatasetExtractRealtime();
    });
}

function startDsExtractRealtime(taskId) {
  closeDatasetExtractRealtime();
  const wsUrl = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/preprocess/tasks/${encodeURIComponent(taskId)}`;
  try {
    state.dsExtractWs = new WebSocket(wsUrl);
    state.dsExtractWs.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      updateDsExtractTaskUi(payload);
      if (["completed", "failed"].includes(payload.status)) closeDatasetExtractRealtime();
    };
    state.dsExtractWs.onerror = () => {
      closeDatasetExtractRealtime();
      state.dsExtractPollTimer = setInterval(() => {
        pollDsExtractTask(taskId).catch(error => appendLog(els.dsExtractLogConsole, error.message));
      }, 1500);
    };
    state.dsExtractWs.onclose = () => {
      if (!state.isUnloading && !["completed", "failed"].includes(state.dsExtractTaskStatus)) {
        state.dsExtractPollTimer = setInterval(() => {
          pollDsExtractTask(taskId).catch(error => appendLog(els.dsExtractLogConsole, error.message));
        }, 1500);
      }
    };
  } catch (error) {
    appendLog(els.dsExtractLogConsole, error.message);
  }
}

async function startDatasetExtract() {
  const ds = datasetById(state.selectedDatasetId);
  if (!ds) throw new Error("请先选择一个数据集。");
  if (!state.dsExtractFile) throw new Error("请先选择待提取的素材文件。");
  if (hasActiveGpuTask()) throw new Error("当前已有活动任务，请等待结束。");

  closeDatasetExtractRealtime();
  state.dsExtractTaskId = null;
  state.dsExtractTaskStatus = "queued";
  state.dsExtractSelection = null;
  state.dsExtractResult = null;
  state.dsExtractConfirmPending = false;
  state.dsExtractDatasetId = ds.id;
  clearMediaElementSource(els.dsExtractOriginalPlayer);
  clearMediaElementSource(els.dsExtractVocalsPlayer);
  els.dsExtractCompareCard?.classList.add("hidden");
  els.dsExtractWarningText?.classList.add("hidden");
  setText(els.dsExtractWarningText, "");
  setText(els.dsExtractStageText, "正在提交提取任务");
  if (els.dsExtractProgressBar) els.dsExtractProgressBar.style.width = "0%";
  setText(els.dsExtractProgressText, "0%");
  if (els.dsExtractLogConsole) els.dsExtractLogConsole.textContent = "等待提取开始…";
  syncDatasetExtractSummary();
  updateDatasetExtractActions();

  const formData = new FormData();
  formData.append("file", state.dsExtractFile);
  formData.append("separator_engine", state.dsExtractEngine || "demucs");
  try {
    const data = await requestJSON(`/api/datasets/${encodeURIComponent(ds.id)}/extract`, { method: "POST", body: formData });
    state.dsExtractTaskId = data.task_id;
    state.dsExtractDatasetId = data.dataset_id || ds.id;
    appendLog(els.dsExtractLogConsole, `[task] 已创建素材提取任务 ${data.task_id}`);
    startDsExtractRealtime(data.task_id);
    toggleBusyActions();
  } catch (error) {
    state.dsExtractTaskStatus = "idle";
    syncDatasetExtractSummary();
    updateDatasetExtractActions();
    toggleBusyActions();
    throw error;
  }
}

async function confirmExtractToDataset() {
  const ds = datasetById(state.selectedDatasetId);
  if (!ds) throw new Error("请先选择一个数据集。");
  if (state.dsExtractDatasetId && state.dsExtractDatasetId !== ds.id) {
    throw new Error("当前提取结果属于其他数据集，请切回原数据集后再确认。");
  }
  if (!state.dsExtractTaskId) throw new Error("还没有可确认的提取结果。");
  if (!state.dsExtractSelection) throw new Error("请先选择要添加的版本。");

  state.dsExtractConfirmPending = true;
  syncDatasetExtractSummary();
  updateDatasetExtractActions();
  appendLog(els.dsExtractLogConsole, `准备将 ${selectedVariantLabel(state.dsExtractSelection)} 添加到数据集…`);

  let successMessage = "";
  try {
    const data = await requestJSON(
      `/api/datasets/${encodeURIComponent(ds.id)}/extract/${encodeURIComponent(state.dsExtractTaskId)}/confirm`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          variant: state.dsExtractSelection,
          auto_segment: state.dsExtractAutoSegment,
        }),
      }
    );
    successMessage = data.segmentation
      ? `素材已添加到数据集，并生成 ${data.segmentation.generated_segments} 条候选片段。`
      : "素材已添加到数据集。";
    resetDatasetExtractUi();
    await refreshOverview().catch(() => {});
    setText(els.dsExtractHintText, successMessage);
    setText(els.datasetUploadHintText, successMessage);
  } finally {
    state.dsExtractConfirmPending = false;
    if (!successMessage) {
      syncDatasetExtractSummary();
    }
    updateDatasetExtractActions();
  }
}

function syncResumeCheckpoints() {
  const detail = versionById(els.trainingResumeVersionSelect?.value);
  state.selectedResumeVersionId = detail?.version?.id || null;
  const ckpts = (detail?.version?.checkpoints || []).map(c => ({
    value: c.id,
    label: `${c.step} steps${c.is_final ? " / final" : ""}`,
  }));
  populateSelect(els.trainingResumeCheckpointSelect, ckpts, ckpts[0]?.value || "", "暂无 checkpoint");
  const mode = currentTrainingMode();
  if (els.trainingResumeVersionLabel) {
    setText(els.trainingResumeVersionLabel, mode === "diffusion_only" ? "扩散目标版本" : "续训基线版本");
  }
  els.trainingResumeCheckpointField?.classList.toggle("hidden", mode !== "resume");
}

function trainingDefaults() {
  const caps = trainingCapabilities();
  const mainDefaults = caps.main_defaults || {};
  const diffusionDefaults = caps.diffusion_defaults || {};
  return {
    steps: state.settings?.training_defaults?.step_count || 2000,
    checkpointInterval: state.settings?.training_defaults?.checkpoint_interval_steps || 500,
    checkpointKeep: state.settings?.training_defaults?.checkpoint_keep_last || 5,
    mainBatchSize: mainDefaults.main_batch_size || mainDefaults.batch_size || 6,
    mainPrecision: mainDefaults.main_precision || "fp32",
    mainAllInMem: Boolean(mainDefaults.main_all_in_mem),
    learningRate: mainDefaults.learning_rate || 0.0001,
    logInterval: mainDefaults.log_interval || 200,
    diffusionMode: diffusionDefaults.diffusion_mode || "disabled",
    diffBatchSize: diffusionDefaults.diff_batch_size || diffusionDefaults.batch_size || 48,
    diffAmpDtype: diffusionDefaults.diff_amp_dtype || diffusionDefaults.amp_dtype || "fp32",
    diffCacheAllData: diffusionDefaults.diff_cache_all_data ?? diffusionDefaults.cache_all_data ?? true,
    diffCacheDevice: diffusionDefaults.diff_cache_device || diffusionDefaults.cache_device || "cpu",
    diffNumWorkers: diffusionDefaults.diff_num_workers ?? diffusionDefaults.num_workers ?? 4,
  };
}

function renderTrainingPresetCards() {
  if (!els.trainingPresetCards) return;
  const presets = trainingPresets();
  if (!presets.find(item => item.id === state.trainingPresetId)) {
    state.trainingPresetId = presets.find(item => item.is_default)?.id || presets[0]?.id || "balanced";
  }
  els.trainingPresetCards.innerHTML = presets.map(preset => {
    const meta = presetMeta(preset.id);
    const disabled = preset.available === false;
    const selected = state.trainingPresetId === preset.id;
    return `
      <button
        class="training-preset-card ${selected ? "is-selected" : ""} ${disabled ? "is-disabled" : ""}"
        type="button"
        data-training-preset="${preset.id}"
        ${disabled ? "disabled" : ""}
      >
        <div class="training-preset-topline">
          <span class="training-preset-mark">${meta.icon}</span>
          <span class="training-preset-name">${preset.label || meta.fallbackLabel}</span>
        </div>
        <div class="training-preset-body">
          <strong>${preset.encoder || "--"}</strong>
          <span>${preset.encoder_dim || "--"} 维</span>
        </div>
        <div class="training-preset-meta">
          <span>${preset.description || "套餐负责架构选择。"} </span>
          <span>推荐显存 ≥ ${preset.recommended_vram_gb || "--"}GB</span>
        </div>
        <div class="training-preset-footline">
          ${preset.is_default ? '<span class="training-preset-badge">推荐</span>' : ""}
          ${disabled ? `<span class="training-preset-disabled">${preset.reason_disabled || "当前不可用"}</span>` : ""}
        </div>
      </button>
    `;
  }).join("");

  const currentPreset = getTrainingPresetById(state.trainingPresetId);
  if (currentPreset) {
    setText(
      els.trainingPresetDerivedText,
      `ssl_dim ${currentPreset.ssl_dim ?? "--"} / gin_channels ${currentPreset.gin_channels ?? "--"} / filter_channels ${currentPreset.filter_channels ?? "--"} 由套餐自动设定`
    );
    setText(
      els.trainingPresetSummaryText,
      `${currentPreset.label || "当前套餐"} 已选中，真正懂的人再展开高级区调运行参数。`
    );
  }
}

function renderReadonlyArchitecture(version) {
  const preset = getTrainingPresetById(inferPresetIdFromVersion(version));
  const presetLabel = version?.main_preset_id ? (preset?.label || version.main_preset_id) : (preset?.label || "原版本架构");
  setText(els.trainingReadonlyPresetText, presetLabel);
  setText(els.trainingReadonlyEncoderText, version?.speech_encoder || preset?.encoder || "--");
  setText(els.trainingReadonlyTinyText, version?.use_tiny ? "已启用" : "未启用");
  setText(els.trainingReadonlySslText, version?.ssl_dim ?? preset?.ssl_dim ?? "--");
  setText(els.trainingReadonlyGinText, version?.gin_channels ?? preset?.gin_channels ?? "--");
  setText(els.trainingReadonlyFilterText, version?.filter_channels ?? preset?.filter_channels ?? "--");
  setText(
    els.trainingReadonlySummaryText,
    version
      ? `这些架构参数来自版本 ${version.label || "--"}，如需更换，请改用完全重训。`
      : "这些架构参数来自原版本配置，如需更换，请改用完全重训。"
  );
}

function applyTrainingDefaults() {
  const defaults = trainingDefaults();
  if (!els.trainingStepsInput?.value) els.trainingStepsInput.value = defaults.steps;
  if (!els.trainingCheckpointIntervalInput?.value) els.trainingCheckpointIntervalInput.value = defaults.checkpointInterval;
  if (!els.trainingCheckpointKeepInput?.value) els.trainingCheckpointKeepInput.value = defaults.checkpointKeep;
  if (!els.trainingMainBatchSizeInput?.value) els.trainingMainBatchSizeInput.value = defaults.mainBatchSize;
  if (!els.trainingLearningRateInput?.value) els.trainingLearningRateInput.value = defaults.learningRate;
  if (!els.trainingLogIntervalInput?.value) els.trainingLogIntervalInput.value = defaults.logInterval;
  if (!els.trainingDiffBatchSizeInput?.value) els.trainingDiffBatchSizeInput.value = defaults.diffBatchSize;
  if (!els.trainingDiffNumWorkersInput?.value) els.trainingDiffNumWorkersInput.value = defaults.diffNumWorkers;
  if (els.trainingMainAllInMemCheckbox && !els.trainingMainAllInMemCheckbox.dataset.seeded) {
    els.trainingMainAllInMemCheckbox.checked = defaults.mainAllInMem;
    els.trainingMainAllInMemCheckbox.dataset.seeded = "1";
  }
  if (els.trainingDiffCacheAllDataCheckbox && !els.trainingDiffCacheAllDataCheckbox.dataset.seeded) {
    els.trainingDiffCacheAllDataCheckbox.checked = Boolean(defaults.diffCacheAllData);
    els.trainingDiffCacheAllDataCheckbox.dataset.seeded = "1";
  }
  if (els.trainingDiffusionModeSelect && !els.trainingDiffusionModeSelect.dataset.seeded) {
    els.trainingDiffusionModeSelect.value = defaults.diffusionMode;
    els.trainingDiffusionModeSelect.dataset.seeded = "1";
  }
  if (els.trainingDiffAmpDtypeSelect && !els.trainingDiffAmpDtypeSelect.dataset.seeded) {
    els.trainingDiffAmpDtypeSelect.value = defaults.diffAmpDtype;
    els.trainingDiffAmpDtypeSelect.dataset.seeded = "1";
  }
  if (els.trainingDiffCacheDeviceSelect && !els.trainingDiffCacheDeviceSelect.dataset.seeded) {
    els.trainingDiffCacheDeviceSelect.value = defaults.diffCacheDevice;
    els.trainingDiffCacheDeviceSelect.dataset.seeded = "1";
  }
}

function renderTrainingConsole() {
  const mode = currentTrainingMode();
  const presets = trainingPresets();
  const dsVersions = state.datasets.flatMap(ds =>
    (ds.versions || []).map(v => ({ value: v.id, label: `${ds.name} / ${v.label}` }))
  );
  populateSelect(els.trainingDatasetVersionSelect, dsVersions, state.selectedDatasetVersionId, "暂无数据集版本");
  if (els.trainingDatasetVersionSelect) {
    els.trainingDatasetVersionSelect.disabled = mode === "diffusion_only";
  }
  populateSelect(els.trainingDeviceSelect, state.runtime?.devices || [], state.runtime?.defaults?.device_preference || "auto", "暂无设备");

  const sourceVersions = state.models.flatMap(m =>
    (m.versions || []).map(v => ({ value: v.id, label: `${m.name} / ${v.label}` }))
  );
  if (!versionById(state.selectedResumeVersionId)) {
    state.selectedResumeVersionId = sourceVersions[0]?.value || null;
  }
  populateSelect(els.trainingResumeVersionSelect, sourceVersions, state.selectedResumeVersionId, "暂无可选版本");
  syncResumeCheckpoints();

  if (!getTrainingPresetById(state.trainingPresetId)) {
    state.trainingPresetId = presets.find(item => item.is_default)?.id || presets[0]?.id || "balanced";
  }
  if (!els.trainingModelNameInput?.value) {
    const ds = datasetById(state.selectedDatasetId);
    if (ds) els.trainingModelNameInput.value = ds.speaker;
  }

  applyTrainingDefaults();

  const sourceVersion = trainingSourceVersionDetail();
  if ((mode === "resume" || mode === "diffusion_only") && sourceVersion) {
    state.trainingPresetId = inferPresetIdFromVersion(sourceVersion);
    if (els.trainingUseTinyCheckbox) {
      els.trainingUseTinyCheckbox.checked = Boolean(sourceVersion.use_tiny);
    }
    renderReadonlyArchitecture(sourceVersion);
  }

  const precisionOptions = precisionOptionsForDevice(els.trainingDeviceSelect?.value || "auto");
  populateSelectWithDisabled(
    els.trainingMainPrecisionSelect,
    precisionOptions,
    els.trainingMainPrecisionSelect?.value || trainingDefaults().mainPrecision,
    "暂无精度选项"
  );

  const tinySupport = tinySupportForPreset(state.trainingPresetId);
  if (els.trainingUseTinyCheckbox) {
    const isLocked = mode === "resume" || mode === "diffusion_only";
    const canUseTiny = !isLocked && tinySupport.available !== false;
    els.trainingUseTinyCheckbox.disabled = !canUseTiny;
    if (!canUseTiny && !isLocked) {
      els.trainingUseTinyCheckbox.checked = false;
    }
    els.trainingUseTinyCheckbox.closest(".advanced-checkline")?.classList.toggle("is-disabled", !canUseTiny || isLocked);
  }
  setText(
    els.trainingUseTinyHintText,
    mode === "resume" || mode === "diffusion_only"
      ? "use_tiny 已继承自原版本配置；如需更换，请改用完全重训"
      : (tinySupport.reason_disabled || "use_tiny：仅在当前套餐支持 tiny 底模时可切")
  );

  const isArchitectureLocked = mode === "resume" || mode === "diffusion_only";
  if (!isArchitectureLocked && els.trainingUseTinyCheckbox) {
    els.trainingUseTinyCheckbox.checked = Boolean(state.trainingUseTiny);
  }
  els.resumeFields?.classList.toggle("hidden", !(mode === "resume" || mode === "diffusion_only"));
  els.trainingPresetSection?.classList.toggle("hidden", isArchitectureLocked);
  els.trainingArchitectureReadonly?.classList.toggle("hidden", !isArchitectureLocked);
  setText(
    els.trainingPresetContextChip,
    isArchitectureLocked ? "当前模式锁定架构" : "套餐决定架构"
  );

  setText(els.trainingTargetModeText, trainingModeLabel(mode));
  setText(
    els.trainingTargetVersionText,
    mode === "diffusion_only"
      ? (sourceVersion?.label || "等待选择扩散目标")
      : mode === "resume"
        ? (sourceVersion?.label || "等待选择续训版本")
        : "新版本"
  );
  setText(
    els.trainingHintText,
    mode === "resume"
      ? "继续训练会继承原版本架构，只允许调整运行参数。"
      : mode === "diffusion_only"
        ? "扩散-only 会保留主模型版本，只补齐扩散模型与状态。数据集沿用目标版本。"
        : "训练基于选定的数据集版本快照，原始数据集可继续编辑。"
  );
  if ((mode === "resume" || mode === "diffusion_only") && sourceVersion && !sourceVersion.dataset_version_id) {
    setText(els.trainingHintText, "该版本没有关联训练数据，请先手动选择数据集版本；如需更换架构，请改用完全重训。");
  }
  setText(
    els.trainingMainRuntimeHintText,
    "Batch Size 与训练精度最影响显存；all_in_mem 更影响内存 / IO，不是主要 VRAM 开关。"
  );
  // Diffusion mode: check asset support for current preset
  const caps = trainingCapabilities();
  const diffAsset = caps.diffusion_asset_support?.[state.trainingPresetId];
  const diffAssetAvailable = diffAsset?.available !== false;
  if (els.trainingDiffusionModeSelect) {
    const isDiffOnly = mode === "diffusion_only";
    els.trainingDiffusionModeSelect.disabled = isDiffOnly || !diffAssetAvailable;
    if (isDiffOnly) {
      els.trainingDiffusionModeSelect.value = "disabled";
    } else if (!diffAssetAvailable) {
      els.trainingDiffusionModeSelect.value = "disabled";
    }
  }
  setText(
    els.trainingDiffusionHintText,
    !diffAssetAvailable
      ? (diffAsset?.reason_disabled || "当前套餐不支持扩散训练。")
      : mode === "diffusion_only"
        ? "扩散-only 模式下扩散训练由目标版本决定。"
        : "启用扩散后训练将花费更多时间与显存。"
  );
  // Diffusion AMP dtype: same device-based disabling as main precision
  if (els.trainingDiffAmpDtypeSelect) {
    const diffPrecisionOptions = precisionOptionsForDevice(els.trainingDeviceSelect?.value || "auto");
    populateSelectWithDisabled(
      els.trainingDiffAmpDtypeSelect,
      diffPrecisionOptions,
      els.trainingDiffAmpDtypeSelect.value || trainingDefaults().diffAmpDtype,
      "暂无精度选项"
    );
  }

  renderTrainingPresetCards();
  renderPredictorCards(
    els.trainingPredictorCards,
    els.trainingPredictorNote,
    state.trainingF0,
    (key) => { state.trainingF0 = key; }
  );
}

function renderModels() {
  setText(els.modelCountChip, String(state.models.length));

  if (!state.models.length) {
    els.modelCards.innerHTML = `
      <div class="empty-state">
        <span class="empty-icon">🧠</span>
        <p class="empty-title">还没有模型</p>
        <p class="hint-text">训练完成后，模型版本会出现在这里。</p>
      </div>`;
    if (els.modelVersionDetail) els.modelVersionDetail.style.display = "none";
    return;
  }

  els.modelCards.innerHTML = state.models.map(m => `
    <button class="model-card ${m.id === state.selectedModelId ? "is-active" : ""}" type="button" data-model-id="${m.id}">
      <div class="card-topline">
        <span class="card-title">${m.name}</span>
        <span class="panel-kicker">${m.speaker}</span>
      </div>
      <div class="card-meta">${(m.versions || []).length} 个版本 · 默认推理 ${m.default_version_id ? "已设定" : "未设定"}</div>
    </button>
  `).join("");
}

function renderModelDetail() {
  const model = modelById(state.selectedModelId);
  if (!model) {
    if (els.modelVersionDetail) els.modelVersionDetail.style.display = "none";
    return;
  }

  if (els.modelVersionDetail) els.modelVersionDetail.style.display = "block";
  setText(els.selectedModelTitle, model.name);
  setText(els.selectedModelMeta, `${model.speaker} · ${(model.versions || []).length} 个版本`);

  els.modelVersionsList.innerHTML = (model.versions || []).length
    ? (model.versions || []).map(v => `
        <article class="version-item ${v.id === state.selectedResumeVersionId ? "is-active" : ""}">
          <div class="card-topline">
            <span class="card-title">${v.label}</span>
            <span class="training-version-mode">${trainingModeLabel(v.training_mode)}</span>
          </div>
          <div class="version-meta">
            <div>F0 ${v.f0_predictor} · ${v.step_count} steps · ${v.device_used || v.device_preference || "--"}</div>
            <div>${v.main_preset_id || inferPresetIdFromVersion(v)} / ${v.speech_encoder || "--"} · ${v.use_tiny ? "tiny" : "standard"} · ${diffusionStatusLabel(v.diffusion_status)}</div>
            <div>${(v.checkpoints || []).slice(0, 5).map(c => `${c.step} steps${c.is_final ? " / final" : ""}`).join(" · ") || "暂无 checkpoint"}</div>
          </div>
          <div class="version-status-row">
            <span class="version-status-chip ${v.diffusion_status === "trained" ? "is-ok" : v.diffusion_status === "training" ? "is-warn" : ""}">${diffusionStatusLabel(v.diffusion_status)}</span>
            ${profileHasDiffusion(v) ? '<span class="version-status-chip is-soft">推理可用扩散</span>' : ""}
          </div>
          <div class="actions compact-actions">
            <button class="ghost-button" type="button" data-model-action="default"  data-model-id="${model.id}" data-version-id="${v.id}">设为推理默认</button>
            <button class="ghost-button" type="button" data-model-action="resume"   data-model-id="${model.id}" data-version-id="${v.id}">继续训练</button>
            <button class="ghost-button" type="button" data-model-action="retrain"  data-model-id="${model.id}" data-version-id="${v.id}">完全重训</button>
            ${v.dataset_version_id && v.diffusion_status !== "training" && v.diffusion_status !== "trained" ? `<button class="ghost-button" type="button" data-model-action="start-diffusion" data-model-id="${model.id}" data-version-id="${v.id}">开始扩散训练</button>` : ""}
          </div>
        </article>
      `).join("")
    : '<p class="hint-text">这个模型还没有版本。</p>';
}

function syncInferenceSpeakers() {
  const profile = (state.runtime?.profiles || []).find(p => p.id === els.inferenceProfileSelect?.value);
  const speakers = profile?.speakers || (profile?.speaker ? [profile.speaker] : []);
  populateSelect(els.inferenceSpeakerSelect, speakers, profile?.speaker || speakers[0] || "", "等待 speaker");
}

function closePreprocessRealtime() {
  if (state.preprocessWs) { state.preprocessWs.close(); state.preprocessWs = null; }
  if (state.preprocessPollTimer) { clearInterval(state.preprocessPollTimer); state.preprocessPollTimer = null; }
}

function setVariantSelection(variant) {
  state.preprocessSelection = variant;
  els.useOriginalButton?.classList.toggle("is-active", variant === "original");
  els.useVocalsButton?.classList.toggle("is-active", variant === "vocals");
  els.preprocessOriginalCard?.classList.toggle("is-selected", variant === "original");
  els.preprocessVocalsCard?.classList.toggle("is-selected", variant === "vocals");
  if (els.preprocessSelectionChip) {
    setText(els.preprocessSelectionChip, variant === "vocals" ? "将用于推理" : variant === "original" ? "使用原始音频" : "未确认");
    els.preprocessSelectionChip.classList.toggle("is-selected", Boolean(variant));
  }
}

function resetPreprocessUi() {
  closePreprocessRealtime();
  state.preprocessTaskId = null;
  state.preprocessTaskStatus = "idle";
  state.preprocessSelection = null;
  state.preprocessResult = null;
  if (els.preprocessAutoCheckbox) {
    state.preprocessAutoExtract = els.preprocessAutoCheckbox.checked;
  }
  if (els.preprocessProgressBar) els.preprocessProgressBar.style.width = "0%";
  setText(els.preprocessProgressText, "0%");
  setText(els.preprocessStageText, "等待选择文件");
  if (els.preprocessLogConsole) els.preprocessLogConsole.textContent = "等待选择文件…";
  setText(els.preprocessStatusChip, "等待文件");
  setText(els.preprocessHintText, "选择文件后，可决定是否先做预处理。");
  setText(els.preprocessWarningText, "");
  els.preprocessWarningText?.classList.add("hidden");
  els.preprocessCompareCard?.classList.add("hidden");
  els.preprocessVocalsCard?.classList.remove("hidden");
  setText(els.preprocessOriginalMeta, "等待结果");
  setText(els.preprocessVocalsMeta, "推荐：含伴奏时优先试听");
  clearMediaElementSource(els.preprocessOriginalPlayer);
  clearMediaElementSource(els.preprocessVocalsPlayer);
  setVariantSelection(null);
}

function updatePreparationLockState() {
  const hasFile = Boolean(state.inferenceFile);
  const needsPreprocess = Boolean(state.preprocessAutoExtract);
  let prepLabel = "等待文件";
  let prepStatusClass = "is-warn";
  let readyLabel = "等待输入";
  let readyHint = "步骤 ② 确认完成后，这里会自动解锁。";

  if (!hasFile) {
    setVariantSelection(null);
  } else if (!needsPreprocess) {
    setVariantSelection("original");
    prepLabel = "可直接推理";
    prepStatusClass = "is-ready";
    readyLabel = "已解锁";
    readyHint = state.inferenceModelLoaded
      ? "当前将直接使用原始音频，参数已可配置。"
      : "输入音频已就绪，加载模型后即可开始推理。";
  } else if (["queued", "running"].includes(state.preprocessTaskStatus)) {
    prepLabel = "提取中…";
    prepStatusClass = "is-warn";
    readyLabel = "等待提取";
    readyHint = "人声分离进行中，完成后会出现对比试听与确认按钮。";
  } else if (state.preprocessTaskStatus === "completed") {
    prepLabel = state.preprocessSelection ? "已确认结果" : "等待确认";
    prepStatusClass = state.preprocessSelection ? "is-ready" : "is-warn";
    readyLabel = state.preprocessSelection ? "已解锁" : "等待确认";
    readyHint = state.preprocessSelection
      ? (state.inferenceModelLoaded ? `当前将使用${selectedVariantLabel(state.preprocessSelection)}进行推理。` : `已确认${selectedVariantLabel(state.preprocessSelection)}，加载模型后即可开始推理。`)
      : "请先试听并点击“使用原始音频”或“使用提取人声”。";
  } else if (state.preprocessTaskStatus === "failed") {
    setVariantSelection("original");
    prepLabel = "提取失败";
    prepStatusClass = "is-warn";
    readyLabel = "已解锁";
    readyHint = state.inferenceModelLoaded
      ? "未检测到清晰人声，将回退使用原始音频继续推理。"
      : "未检测到清晰人声，加载模型后将回退使用原始音频继续推理。";
  } else if (hasFile) {
    prepLabel = "等待提取";
    prepStatusClass = "is-warn";
    readyLabel = "等待预处理";
    readyHint = "已选择文件，如需分离请点击“提取人声”。";
  }

  setText(els.preprocessStatusChip, prepLabel);
  els.preprocessStatusChip?.classList.toggle("is-ready", prepStatusClass === "is-ready");
  els.preprocessStatusChip?.classList.toggle("is-warn", prepStatusClass === "is-warn");

  const step3Ready = isPreprocessReadySelection();
  setText(els.inferenceReadyChip, step3Ready ? "已解锁" : readyLabel);
  els.inferenceReadyChip?.classList.toggle("is-ready", step3Ready);
  els.inferenceReadyChip?.classList.toggle("is-warn", !step3Ready && hasFile);
  els.inferenceReadyChip?.classList.toggle("is-locked", !step3Ready);
  setText(els.inferenceReadyHintText, readyHint);
  els.inferenceRunPanel?.classList.toggle("is-locked", !step3Ready || !state.inferenceModelLoaded);
  if (els.startInferenceButton) {
    const busy = hasActiveGpuTask();
    els.startInferenceButton.disabled = !step3Ready || !state.inferenceModelLoaded || busy;
  }

  setText(els.selectedInputVariantText, selectedVariantLabel(state.preprocessSelection));
  if (!hasFile) {
    setText(els.selectedInputSourceText, "等待准备");
  } else if (state.preprocessTaskStatus === "completed" && state.preprocessSelection === "vocals") {
    setText(els.selectedInputSourceText, `来自 ${engineLabel(state.preprocessEngine)} 的提取结果`);
  } else if (state.preprocessTaskStatus === "completed" && state.preprocessSelection === "original") {
    setText(els.selectedInputSourceText, "已试听并选择原始音频");
  } else if (state.preprocessTaskStatus === "failed") {
    setText(els.selectedInputSourceText, "提取失败，回退原始音频");
  } else if (canAdvanceWithoutPreprocess()) {
    setText(els.selectedInputSourceText, "未启用人声分离");
  } else {
    setText(els.selectedInputSourceText, "等待预处理确认");
  }
}

function updatePreprocessActions() {
  const hasFile = Boolean(state.inferenceFile);
  const busy = hasActiveGpuTask();
  const running = ["queued", "running"].includes(state.preprocessTaskStatus);
  const needsPreprocess = Boolean(state.preprocessAutoExtract);

  if (els.startPreprocessButton) {
    els.startPreprocessButton.disabled = !hasFile || !needsPreprocess || busy;
    els.startPreprocessButton.textContent = running ? "提取中…" : "提取人声";
  }

  const canChooseOriginal = state.preprocessTaskStatus === "completed" || state.preprocessTaskStatus === "failed";
  const canChooseVocals = state.preprocessTaskStatus === "completed";
  if (els.useOriginalButton) els.useOriginalButton.disabled = !canChooseOriginal || running || busy;
  if (els.useVocalsButton) els.useVocalsButton.disabled = !canChooseVocals || running || busy;
  if (els.preprocessAutoCheckbox) els.preprocessAutoCheckbox.disabled = running || busy;
}

function updatePreprocessComparisonUi() {
  const completed = state.preprocessTaskStatus === "completed";
  const failed = state.preprocessTaskStatus === "failed";
  const hasResult = completed || failed;
  els.preprocessCompareCard?.classList.toggle("hidden", !hasResult);
  els.preprocessVocalsCard?.classList.toggle("hidden", failed);
  if (!hasResult) {
    clearMediaElementSource(els.preprocessOriginalPlayer);
    clearMediaElementSource(els.preprocessVocalsPlayer);
    return;
  }

  if (completed && state.preprocessResult?.original_url) {
    els.preprocessOriginalPlayer.src = state.preprocessResult.original_url;
    els.preprocessOriginalPlayer.load();
  } else if (failed && state.preprocessResult?.original_url) {
    els.preprocessOriginalPlayer.src = state.preprocessResult.original_url;
    els.preprocessOriginalPlayer.load();
  } else {
    clearMediaElementSource(els.preprocessOriginalPlayer);
  }

  if (completed && state.preprocessResult?.vocals_url) {
    els.preprocessVocalsPlayer.src = state.preprocessResult.vocals_url;
    els.preprocessVocalsPlayer.load();
  } else {
    clearMediaElementSource(els.preprocessVocalsPlayer);
  }
}

function updatePreprocessTaskUi(payload) {
  const summary = payload.summary || {};
  const activeEngine = summary.separator_engine || payload.separator_engine || state.preprocessEngine;
  const progress = Math.max(0, Math.min(100, Number(payload.progress ?? 0)));
  state.preprocessTaskStatus = payload.status || state.preprocessTaskStatus;
  state.preprocessResult = payload;
  setText(els.preprocessStageText, `${payload.stage || payload.status || "--"} · ${payload.current_segment ?? "--"} / ${payload.total_segments ?? "--"}`);
  if (els.preprocessProgressBar) els.preprocessProgressBar.style.width = `${progress}%`;
  setText(els.preprocessProgressText, `${progress.toFixed(0)}%`);
  if (payload.message) appendLog(els.preprocessLogConsole, payload.message);

  if (["queued", "running"].includes(payload.status)) {
    clearMediaElementSource(els.preprocessOriginalPlayer);
    clearMediaElementSource(els.preprocessVocalsPlayer);
    els.preprocessCompareCard?.classList.add("hidden");
    setVariantSelection(null);
    els.preprocessWarningText?.classList.add("hidden");
    setText(els.preprocessWarningText, "");
  }

  if (payload.status === "completed") {
    setText(els.preprocessOriginalMeta, payload.original_label || "原始音轨");
    setText(els.preprocessVocalsMeta, payload.vocals_label || `来自 ${engineLabel(activeEngine)} 的提取结果`);
    updatePreprocessComparisonUi();
  }

  if (payload.status === "failed") {
    setText(els.preprocessOriginalMeta, payload.original_label || "原始音轨");
    setText(els.preprocessWarningText, payload.warning || "未检测到清晰人声，建议直接使用原始音频。");
    els.preprocessWarningText?.classList.remove("hidden");
    updatePreprocessComparisonUi();
    setVariantSelection("original");
  }

  updatePreparationLockState();
  updatePreprocessActions();
  toggleBusyActions();
}

function pollPreprocessTask(taskId) {
  return requestJSON(`/api/preprocess/tasks/${encodeURIComponent(taskId)}`)
    .then(payload => {
      updatePreprocessTaskUi(payload);
      if (["completed", "failed"].includes(payload.status)) closePreprocessRealtime();
    });
}

function startPreprocessRealtime(taskId) {
  closePreprocessRealtime();
  const wsUrl = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/preprocess/tasks/${encodeURIComponent(taskId)}`;
  try {
    state.preprocessWs = new WebSocket(wsUrl);
    state.preprocessWs.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      updatePreprocessTaskUi(payload);
      if (["completed", "failed"].includes(payload.status)) closePreprocessRealtime();
    };
    state.preprocessWs.onerror = () => {
      closePreprocessRealtime();
      state.preprocessPollTimer = setInterval(() => {
        pollPreprocessTask(taskId).catch(error => appendLog(els.preprocessLogConsole, error.message));
      }, 1500);
    };
    state.preprocessWs.onclose = () => {
      if (!state.isUnloading && !["completed", "failed"].includes(state.preprocessTaskStatus)) {
        state.preprocessPollTimer = setInterval(() => {
          pollPreprocessTask(taskId).catch(error => appendLog(els.preprocessLogConsole, error.message));
        }, 1500);
      }
    };
  } catch (error) {
    appendLog(els.preprocessLogConsole, error.message);
  }
}

async function startPreprocess() {
  if (!state.inferenceFile) throw new Error("请先选择待处理的音频或视频文件。");
  if (!state.preprocessAutoExtract) throw new Error("请先开启“自动提取人声”。");
  if (hasActiveGpuTask()) throw new Error("当前已有活动任务，请等待结束。");

  closePreprocessRealtime();
  state.preprocessTaskId = null;
  state.preprocessTaskStatus = "queued";
  state.preprocessSelection = null;
  state.preprocessResult = null;
  clearMediaElementSource(els.preprocessOriginalPlayer);
  clearMediaElementSource(els.preprocessVocalsPlayer);
  els.preprocessCompareCard?.classList.add("hidden");
  els.preprocessWarningText?.classList.add("hidden");
  setText(els.preprocessWarningText, "");
  setText(els.preprocessStageText, "正在提交预处理任务");
  if (els.preprocessProgressBar) els.preprocessProgressBar.style.width = "0%";
  setText(els.preprocessProgressText, "0%");
  if (els.preprocessLogConsole) els.preprocessLogConsole.textContent = "等待预处理开始…";
  updatePreparationLockState();
  updatePreprocessActions();

  const formData = new FormData();
  formData.append("file", state.inferenceFile);
  formData.append("separator_engine", state.preprocessEngine || "demucs");
  try {
    const data = await requestJSON("/api/preprocess/tasks", { method: "POST", body: formData });
    state.preprocessTaskId = data.task_id;
    appendLog(els.preprocessLogConsole, `[task] 已创建预处理任务 ${data.task_id}`);
    startPreprocessRealtime(data.task_id);
    toggleBusyActions();
  } catch (error) {
    state.preprocessTaskStatus = "idle";
    updatePreparationLockState();
    updatePreprocessActions();
    toggleBusyActions();
    throw error;
  }
}

function renderInferenceWorkbench() {
  const rt = state.runtime || {};
  const profiles = (rt.profiles || []).map(p => ({ value: p.id, label: p.label }));
  if (!profiles.find(p => p.value === state.selectedInferenceProfileId))
    state.selectedInferenceProfileId = rt.defaults?.profile_id || profiles[0]?.value || null;

  populateSelect(els.inferenceProfileSelect, profiles, state.selectedInferenceProfileId, "暂无推理版本");
  state.selectedInferenceProfileId = els.inferenceProfileSelect?.value || state.selectedInferenceProfileId;

  populateSelect(els.inferenceDeviceSelect, rt.devices || [], rt.defaults?.device_preference || "auto", "暂无设备");

  syncInferenceSpeakers();
  const profile = selectedInferenceProfile();
  if (!profileHasDiffusion(profile)) {
    state.inferenceUseDiffusion = false;
  }
  if (els.inferenceUseDiffusionCheckbox) {
    els.inferenceUseDiffusionCheckbox.checked = state.inferenceUseDiffusion;
  }

  const availableSeparatorOptions = (rt.separator_engines || []).filter(option => option.available !== false);
  if (availableSeparatorOptions.length && !availableSeparatorOptions.find(option => option.value === state.preprocessEngine)) {
    state.preprocessEngine = availableSeparatorOptions[0].value;
  }

  if (els.acceptedMediaText) {
    setText(els.acceptedMediaText, rt.accepted_media_types?.join(" / ") || "支持 WAV / MP3 / MP4 / MKV / MOV / AVI");
  }
  if (els.preprocessAutoCheckbox) {
    els.preprocessAutoCheckbox.checked = state.preprocessAutoExtract;
  }

  // F0 卡片
  renderPredictorCards(
    els.inferencePredictorCards,
    els.inferencePredictorNote,
    state.inferenceF0,
    (key) => { state.inferenceF0 = key; }
  );

  renderSeparatorCards(
    els.separatorEngineCards,
    els.separatorEngineNote,
    state.preprocessEngine,
    (key) => { state.preprocessEngine = key; }
  );

  // 更新加载状态 UI
  updateInferenceLoadUi();
  updatePreparationLockState();
  updatePreprocessActions();
}

function updateInferenceLoadUi() {
  const loaded = state.inferenceModelLoaded;
  const profile = selectedInferenceProfile();
  const hasDiffusion = profileHasDiffusion(profile);
  setText(els.inferenceLoadStatusChip, loaded ? "✓ 已加载" : "未加载");
  if (els.inferenceLoadStatusChip) {
    els.inferenceLoadStatusChip.classList.toggle("is-loaded", loaded);
  }
  if (els.inferenceUseDiffusionCheckbox) {
    els.inferenceUseDiffusionCheckbox.checked = hasDiffusion ? state.inferenceUseDiffusion : false;
    els.inferenceUseDiffusionCheckbox.disabled = !hasDiffusion;
  }
  if (els.inferenceDiffusionCard) {
    els.inferenceDiffusionCard.style.display = hasDiffusion ? "" : "none";
  }
  setText(
    els.inferenceDiffusionHintText,
    hasDiffusion
      ? (state.inferenceUseDiffusion ? "已选择扩散增强；切换后需要重新加载模型。" : "当前版本带有扩散模型，可按需开启。")
      : "当前版本没有可用扩散模型；切换版本后会自动刷新这里。"
  );
  if (els.unloadInferenceModelButton) {
    const busy = hasActiveGpuTask();
    els.unloadInferenceModelButton.disabled = !loaded || busy;
  }
  updatePreparationLockState();
}

function toggleBusyActions() {
  const busy = hasActiveGpuTask();
  const btns = [
    els.createDatasetButton, els.uploadDatasetButton, els.rerunSegmentButton,
    els.createDatasetVersionButton, els.startTrainingButton,
    els.loadInferenceModelButton, els.unloadInferenceModelButton, els.startInferenceButton, els.startPreprocessButton,
    els.dsExtractStartButton, els.dsExtractConfirmButton,
  ];
  btns.forEach(btn => {
    if (!btn) return;
    btn.disabled = busy;
    btn.classList.toggle("disabled", busy);
  });
  updateDatasetExtractActions();
  updatePreprocessActions();
  updateInferenceLoadUi();
}

// ── 数据集操作 ───────────────────────────────────────────────────────────────
async function createDataset() {
  const speaker = els.datasetSpeakerInput.value.trim();
  if (!speaker) throw new Error("创建数据集至少要填 Speaker 名称。");
  const data = await requestJSON("/api/datasets", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: els.datasetNameInput.value.trim() || speaker,
      speaker,
      description: els.datasetDescriptionInput.value.trim(),
    }),
  });
  state.selectedDatasetId = data.dataset.id;
  resetDatasetExtractUi();
  setText(els.datasetUploadHintText, "数据集已创建，可以直接上传原始录音。");
  await refreshOverview();
}

async function uploadDatasetFiles() {
  const ds = datasetById(state.selectedDatasetId);
  if (!ds) throw new Error("先创建或选择一个数据集。");
  const files = Array.from(els.datasetUploadInput.files || []);
  if (!files.length) throw new Error("请先选择训练音频文件。");
  const formData = new FormData();
  files.forEach(f => formData.append("files", f));
  formData.append("auto_segment", els.datasetAutoSegmentCheckbox.checked ? "true" : "false");
  setText(els.datasetUploadHintText, "上传中，请稍候…");
  const data = await requestJSON(`/api/datasets/${ds.id}/files`, { method: "POST", body: formData });
  setText(els.datasetUploadHintText,
    data.segmentation
      ? `上传成功，已生成 ${data.segmentation.generated_segments} 条候选片段。`
      : "上传成功。"
  );
  await refreshOverview();
}

async function rerunSegmentize() {
  const ds = datasetById(state.selectedDatasetId);
  if (!ds) throw new Error("先选择一个数据集。");
  setText(els.datasetUploadHintText, "重新分段中，请稍候…");
  const result = await requestJSON(`/api/datasets/${ds.id}/segmentize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ max_segment_seconds: 6.0, min_keep_seconds: 1.5, merge_gap_ms: 300, energy_floor_db: -45 }),
  });
  setText(els.datasetUploadHintText, `已重新分段，共生成 ${result.generated_segments} 条候选片段。`);
  await refreshOverview();
}

async function toggleSegment(segmentId, enabled) {
  await requestJSON(`/api/segments/${segmentId}/enabled`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
  await refreshOverview();
}

async function createDatasetVersion() {
  const ds = datasetById(state.selectedDatasetId);
  if (!ds) throw new Error("先选择一个数据集。");
  const label = els.datasetVersionLabelInput.value.trim();
  if (!label) throw new Error("请填写版本标签（例如 v2_clean）。");
  const data = await requestJSON(`/api/datasets/${ds.id}/versions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label, notes: "由前端工作台创建的数据集版本。" }),
  });
  state.selectedDatasetVersionId = data.dataset_version.id;
  els.datasetVersionLabelInput.value = "";
  await refreshOverview();
}

// ── 设置操作 ─────────────────────────────────────────────────────────────────
async function saveSettings() {
  await requestJSON("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      app: { data_root: els.settingsDataRootInput.value.trim() || "workspace_data" },
      training_defaults: {
        f0_predictor: els.settingsDefaultF0Select.value,
        step_count: Number(els.settingsDefaultStepsInput.value || 2000),
        checkpoint_interval_steps: Number(els.settingsCheckpointIntervalInput.value || 500),
        checkpoint_keep_last: Number(els.settingsCheckpointKeepInput.value || 5),
      },
    }),
  });
  setText(els.settingsHintText, "✓ 设置已保存。");
  await refreshOverview();
}

// ── 训练 ─────────────────────────────────────────────────────────────────────
function resetTrainingUi() {
  state.trainingTaskStatus = "idle";
  setText(els.trainingTaskStatusText, "idle");
  setText(els.trainingStageText, "等待训练开始");
  if (els.trainingProgressBar) els.trainingProgressBar.style.width = "0%";
  setText(els.trainingProgressText, "0%");
  if (els.trainingLogConsole) els.trainingLogConsole.textContent = "等待训练开始…";
  setText(els.trainingResultVersionText, "--");
  setText(els.trainingResultCheckpointText, "--");
  setText(els.trainingResultDeviceText, "--");
  els.trainingCompleteCard?.classList.add("hidden");
}

function updateTrainingTaskUi(payload) {
  const summary = payload.summary || {};
  const progress = Math.max(0, Math.min(100, Number(payload.progress ?? 0)));
  state.trainingTaskStatus = payload.status || state.trainingTaskStatus;

  setText(els.trainingTaskStatusText, payload.status || "running");
  setText(els.trainingStageText, `${payload.stage || payload.status || "--"} · ${payload.current_segment ?? "--"} / ${payload.total_segments ?? "--"}`);
  if (els.trainingProgressBar) els.trainingProgressBar.style.width = `${progress}%`;
  setText(els.trainingProgressText, `${progress.toFixed(0)}%`);
  if (payload.message) appendLog(els.trainingLogConsole, payload.message);
  if (summary.registered_label) setText(els.trainingResultVersionText, summary.registered_label);
  if (summary.latest_checkpoint) setText(els.trainingResultCheckpointText, summary.latest_checkpoint);
  if (summary.device_used || summary.feature_device_used)
    setText(els.trainingResultDeviceText, summary.device_used || summary.feature_device_used);

  if (payload.status === "completed") {
    // 自动更新模型库状态
    if (summary.model_id) state.selectedModelId = summary.model_id;
    if (summary.model_version_id) state.selectedInferenceProfileId = summary.model_version_id;

    // 显示训练完成引导卡片
    if (els.trainingCompleteCard) {
      els.trainingCompleteCard.classList.remove("hidden");
      setText(els.trainingCompleteTitle, `训练完成 — ${summary.registered_label || "新版本"}`);
      setText(els.trainingCompleteMeta, summary.latest_checkpoint ? `最新 checkpoint：${summary.latest_checkpoint}` : "");
    }
  }

  toggleBusyActions();
  updatePipelineState();
}

function closeTrainingRealtime() {
  if (state.trainingWs) { state.trainingWs.close(); state.trainingWs = null; }
  if (state.trainingPollTimer) { clearInterval(state.trainingPollTimer); state.trainingPollTimer = null; }
}

function startTrainingRealtime(taskId) {
  closeTrainingRealtime();
  const wsUrl = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/training/tasks/${encodeURIComponent(taskId)}`;
  try {
    state.trainingWs = new WebSocket(wsUrl);
    state.trainingWs.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      updateTrainingTaskUi(payload);
      if (["completed", "failed"].includes(payload.status)) {
        closeTrainingRealtime();
        refreshOverview().then(() => {
          renderModels();
          renderModelDetail();
          renderInferenceWorkbench();
        }).catch(() => {});
      }
    };
    state.trainingWs.onerror = () => {
      closeTrainingRealtime();
      state.trainingPollTimer = setInterval(() => {
        requestJSON(`/api/training/tasks/${encodeURIComponent(taskId)}`)
          .then(p => {
            updateTrainingTaskUi(p);
            if (["completed", "failed"].includes(p.status)) {
              closeTrainingRealtime();
              refreshOverview().catch(() => {});
            }
          })
          .catch(e => appendLog(els.trainingLogConsole, e.message));
      }, 1500);
    };
    state.trainingWs.onclose = () => {
      if (!state.isUnloading && !["completed", "failed"].includes(state.trainingTaskStatus)) {
        state.trainingPollTimer = setInterval(() => {
          requestJSON(`/api/training/tasks/${encodeURIComponent(taskId)}`)
            .then(p => {
              updateTrainingTaskUi(p);
              if (["completed", "failed"].includes(p.status)) closeTrainingRealtime();
            })
            .catch(e => appendLog(els.trainingLogConsole, e.message));
        }, 1500);
      }
    };
  } catch (e) {
    appendLog(els.trainingLogConsole, e.message);
  }
}

async function startTraining() {
  if (hasActiveGpuTask())
    throw new Error("当前已有活动任务，请等待结束。");
  const mode = currentTrainingMode();
  const datasetVersionId = els.trainingDatasetVersionSelect?.value;
  if (!datasetVersionId) throw new Error("请先选择一个数据集版本。");
  if (!els.trainingModelNameInput?.value.trim()) throw new Error("请先填写模型名称。");
  if (mode === "resume" && !els.trainingResumeCheckpointSelect?.value)
    throw new Error("继续训练需要选择一个 checkpoint。");
  if (mode === "diffusion_only" && !els.trainingResumeVersionSelect?.value)
    throw new Error("请先选择一个扩散目标版本。");

  resetTrainingUi();

  const formData = new FormData();
  formData.append("dataset_version_id", datasetVersionId);
  formData.append("model_name", els.trainingModelNameInput.value.trim());
  formData.append("device_preference", els.trainingDeviceSelect?.value || "auto");
  formData.append("f0_predictor", state.trainingF0 || "rmvpe");
  formData.append("max_steps", els.trainingStepsInput?.value || String(state.settings?.training_defaults?.step_count || 2000));
  formData.append("training_mode", mode);
  formData.append("resume_from_checkpoint_id", els.trainingResumeCheckpointSelect?.value || "");
  formData.append("checkpoint_interval_steps", els.trainingCheckpointIntervalInput?.value || String(state.settings?.training_defaults?.checkpoint_interval_steps || 500));
  formData.append("checkpoint_keep_last", els.trainingCheckpointKeepInput?.value || String(state.settings?.training_defaults?.checkpoint_keep_last || 5));
  formData.append("main_preset_id", mode === "new" ? (state.trainingPresetId || "balanced") : "");
  formData.append("main_batch_size", els.trainingMainBatchSizeInput?.value || String(trainingDefaults().mainBatchSize));
  formData.append("main_precision", els.trainingMainPrecisionSelect?.value || trainingDefaults().mainPrecision);
  formData.append("main_all_in_mem", els.trainingMainAllInMemCheckbox?.checked ? "true" : "false");
  formData.append(
    "use_tiny",
    mode === "new"
      ? (els.trainingUseTinyCheckbox?.checked ? "true" : "false")
      : (trainingSourceVersionDetail()?.use_tiny ? "true" : "false")
  );
  formData.append("learning_rate", els.trainingLearningRateInput?.value || String(trainingDefaults().learningRate));
  formData.append("log_interval", els.trainingLogIntervalInput?.value || String(trainingDefaults().logInterval));
  formData.append("diffusion_mode", mode === "diffusion_only" ? "disabled" : (els.trainingDiffusionModeSelect?.value || "disabled"));
  formData.append("diff_batch_size", els.trainingDiffBatchSizeInput?.value || String(trainingDefaults().diffBatchSize));
  formData.append("diff_amp_dtype", els.trainingDiffAmpDtypeSelect?.value || trainingDefaults().diffAmpDtype);
  formData.append("diff_cache_all_data", els.trainingDiffCacheAllDataCheckbox?.checked ? "true" : "false");
  formData.append("diff_cache_device", els.trainingDiffCacheDeviceSelect?.value || trainingDefaults().diffCacheDevice);
  formData.append("diff_num_workers", els.trainingDiffNumWorkersInput?.value || String(trainingDefaults().diffNumWorkers));
  formData.append("target_model_version_id", mode === "diffusion_only" ? (els.trainingResumeVersionSelect?.value || "") : "");

  const data = await requestJSON("/api/training/tasks", { method: "POST", body: formData });
  state.trainingTaskId = data.task_id;
  state.trainingTaskStatus = "queued";
  appendLog(els.trainingLogConsole, `[task] 已创建训练任务 ${data.task_id}`);
  startTrainingRealtime(data.task_id);
  toggleBusyActions();
}

async function performModelAction(action, modelId, versionId) {
  const detail = versionById(versionId);
  if (!detail) return;

  if (action === "default") {
    await requestJSON(`/api/models/${modelId}/default-version`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_version_id: versionId }),
    });
    await refreshOverview();
    return;
  }

  state.selectedResumeVersionId = versionId;
  state.selectedDatasetVersionId = detail.version.dataset_version_id || state.selectedDatasetVersionId;
  state.trainingPresetId = inferPresetIdFromVersion(detail.version);
  state.trainingUseTiny = Boolean(detail.version.use_tiny);
  if (els.trainingModelNameInput) els.trainingModelNameInput.value = detail.model.name;
  if (els.trainingModeSelect) {
    if (action === "resume") {
      els.trainingModeSelect.value = "resume";
    } else if (action === "start-diffusion") {
      els.trainingModeSelect.value = "diffusion_only";
    } else {
      els.trainingModeSelect.value = "new";
    }
  }
  if (els.trainingUseTinyCheckbox) {
    els.trainingUseTinyCheckbox.checked = Boolean(detail.version.use_tiny);
  }
  if (action === "start-diffusion") {
    setText(els.trainingHintText, "已根据版本预填扩散-only 任务，确认参数后再启动。");
  }
  renderTrainingConsole();
  setActiveView("training");
}

// ── 推理 ─────────────────────────────────────────────────────────────────────
async function loadInferenceModel() {
  if (hasActiveGpuTask()) {
    throw new Error("当前已有活动任务，请等待结束。");
  }
  const profile = selectedInferenceProfile();
  if (!profile) throw new Error("请先选择一个推理模型版本。");
  setText(els.inferenceModelHintText, "加载中，请稍候…");
  await requestJSON("/api/runtime/load-model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      profile_id: profile.id,
      speaker: els.inferenceSpeakerSelect?.value || profile.speaker || "",
      device_preference: els.inferenceDeviceSelect?.value || "auto",
      use_diffusion: state.inferenceUseDiffusion,
    }),
  });
  state.inferenceModelLoaded = true;
  setText(
    els.inferenceModelHintText,
    `✓ 已加载：${profile.label || profile.id}${state.inferenceUseDiffusion && profileHasDiffusion(profile) ? "（扩散增强已启用）" : ""}`
  );
  updateInferenceLoadUi();
  await refreshOverview();
}

async function unloadInferenceModel() {
  if (hasActiveGpuTask()) {
    throw new Error("当前有活动任务，无法卸载模型。");
  }
  await requestJSON("/api/runtime/unload-model", { method: "POST" });
  state.inferenceModelLoaded = false;
  setText(els.inferenceModelHintText, "模型已卸载，输入文件和预处理结果已保留，重新加载后即可继续。");
  updateInferenceLoadUi();
  await refreshOverview();
}

function resetInferenceUi() {
  state.inferenceTaskStatus = "idle";
  setText(els.inferenceTaskStatusText, "idle");
  setText(els.inferenceStageText, "等待推理开始");
  if (els.inferenceProgressBar) els.inferenceProgressBar.style.width = "0%";
  setText(els.inferenceProgressText, "0%");
  if (els.inferenceLogConsole) els.inferenceLogConsole.textContent = "等待推理开始…";
  if (els.inferenceResultPlayer) els.inferenceResultPlayer.removeAttribute("src");
  if (els.inferenceDownloadLink) {
    els.inferenceDownloadLink.classList.add("disabled");
    els.inferenceDownloadLink.removeAttribute("href");
  }
}

function updateInferenceTaskUi(payload) {
  const progress = Math.max(0, Math.min(100, Number(payload.progress ?? 0)));
  state.inferenceTaskStatus = payload.status || state.inferenceTaskStatus;
  setText(els.inferenceTaskStatusText, payload.status || "running");
  setText(els.inferenceStageText, `${payload.stage || payload.status || "--"} · ${payload.current_segment ?? "--"} / ${payload.total_segments ?? "--"}`);
  if (els.inferenceProgressBar) els.inferenceProgressBar.style.width = `${progress}%`;
  setText(els.inferenceProgressText, `${progress.toFixed(0)}%`);
  if (payload.message) appendLog(els.inferenceLogConsole, payload.message);
  if (payload.result_url) {
    if (els.inferenceResultPlayer) els.inferenceResultPlayer.src = payload.result_url;
    if (els.inferenceDownloadLink) {
      els.inferenceDownloadLink.href = payload.result_url;
      els.inferenceDownloadLink.classList.remove("disabled");
    }
  }
  toggleBusyActions();
  updatePipelineState();
}

function closeInferenceRealtime() {
  if (state.inferenceWs) { state.inferenceWs.close(); state.inferenceWs = null; }
  if (state.inferencePollTimer) { clearInterval(state.inferencePollTimer); state.inferencePollTimer = null; }
}

function startInferenceRealtime(taskId) {
  closeInferenceRealtime();
  const wsUrl = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/tasks/${encodeURIComponent(taskId)}`;
  try {
    state.inferenceWs = new WebSocket(wsUrl);
    state.inferenceWs.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      updateInferenceTaskUi(payload);
      if (["completed", "failed"].includes(payload.status)) closeInferenceRealtime();
    };
    state.inferenceWs.onerror = () => {
      closeInferenceRealtime();
      state.inferencePollTimer = setInterval(() => {
        requestJSON(`/api/tasks/${encodeURIComponent(taskId)}`)
          .then(p => { updateInferenceTaskUi(p); if (["completed", "failed"].includes(p.status)) closeInferenceRealtime(); })
          .catch(e => appendLog(els.inferenceLogConsole, e.message));
      }, 1500);
    };
    state.inferenceWs.onclose = () => {
      if (!state.isUnloading && !["completed", "failed"].includes(state.inferenceTaskStatus)) {
        state.inferencePollTimer = setInterval(() => {
          requestJSON(`/api/tasks/${encodeURIComponent(taskId)}`)
            .then(p => { updateInferenceTaskUi(p); if (["completed", "failed"].includes(p.status)) closeInferenceRealtime(); })
            .catch(e => appendLog(els.inferenceLogConsole, e.message));
        }, 1500);
      }
    };
  } catch (e) {
    appendLog(els.inferenceLogConsole, e.message);
  }
}

async function startInference() {
  if (hasActiveGpuTask())
    throw new Error("当前已有活动任务，请等待结束。");
  if (!state.inferenceFile) throw new Error("请先选择待处理的音频或视频文件。");
  if (!state.inferenceModelLoaded)
    throw new Error("请先加载推理模型（步骤 1）。");
  if (!isPreprocessReadySelection())
    throw new Error("请先在步骤 ② 选择“使用原始音频”或“使用提取人声”。");

  resetInferenceUi();

  const formData = new FormData();
  formData.append("profile_id", els.inferenceProfileSelect?.value || "");
  formData.append("speaker", els.inferenceSpeakerSelect?.value || "");
  formData.append("device_preference", els.inferenceDeviceSelect?.value || "auto");
  formData.append("use_diffusion", state.inferenceUseDiffusion ? "true" : "false");
  formData.append("tran", els.inferenceTranInput?.value || "0");
  formData.append("slice_db", els.inferenceSliceInput?.value || "-40");
  formData.append("noise_scale", els.inferenceNoiseInput?.value || "0.4");
  formData.append("pad_seconds", els.inferencePadInput?.value || "0.5");
  formData.append("f0_predictor", state.inferenceF0 || "rmvpe");

  if (state.preprocessTaskId && state.preprocessResult && isPreprocessReadySelection()) {
    formData.append("prepared_task_id", state.preprocessTaskId);
    formData.append("prepared_variant", state.preprocessSelection);
  } else {
    formData.append("file", state.inferenceFile);
  }

  const data = await requestJSON("/api/tasks", { method: "POST", body: formData });
  state.inferenceTaskId = data.task_id;
  state.inferenceTaskStatus = "queued";
  appendLog(els.inferenceLogConsole, `[task] 已创建推理任务 ${data.task_id}`);
  startInferenceRealtime(data.task_id);
  toggleBusyActions();
}

// ── 事件绑定 ─────────────────────────────────────────────────────────────────
function bindEvents() {
  // Tab 导航
  bindTabNav();

  // Settings 开关
  els.settingsTrigger?.addEventListener("click", openSettings);
  els.settingsCloseBtn?.addEventListener("click", closeSettings);
  els.settingsOverlay?.addEventListener("click", closeSettings);

  // Settings 操作
  els.saveSettingsButton?.addEventListener("click", () =>
    saveSettings().catch(e => setText(els.settingsHintText, `错误：${e.message}`))
  );
  els.refreshAllButton?.addEventListener("click", () =>
    refreshOverview().catch(e => setText(els.settingsHintText, `错误：${e.message}`))
  );

  // 数据集
  els.createDatasetButton?.addEventListener("click", () =>
    createDataset().catch(e => setText(els.datasetUploadHintText, `错误：${e.message}`))
  );
  els.uploadDatasetButton?.addEventListener("click", () =>
    uploadDatasetFiles().catch(e => setText(els.datasetUploadHintText, `错误：${e.message}`))
  );
  els.rerunSegmentButton?.addEventListener("click", () =>
    rerunSegmentize().catch(e => setText(els.datasetUploadHintText, `错误：${e.message}`))
  );
  els.createDatasetVersionButton?.addEventListener("click", () =>
    createDatasetVersion().catch(e => setText(els.datasetUploadHintText, `错误：${e.message}`))
  );
  els.dsExtractFileInput?.addEventListener("change", (e) => {
    const file = e.target.files?.[0] || null;
    state.dsExtractFile = file;
    state.dsExtractDatasetId = state.selectedDatasetId;
    if (file) {
      resetDatasetExtractUi({ preserveFile: true });
    } else {
      resetDatasetExtractUi();
    }
  });
  els.dsExtractAutoSegmentCheckbox?.addEventListener("change", () => {
    state.dsExtractAutoSegment = Boolean(els.dsExtractAutoSegmentCheckbox.checked);
  });
  els.dsExtractStartButton?.addEventListener("click", () =>
    startDatasetExtract().catch(e => {
      syncDatasetExtractSummary();
      appendLog(els.dsExtractLogConsole, `错误：${e.message}`);
    })
  );
  els.dsExtractUseOriginalButton?.addEventListener("click", () => {
    if (els.dsExtractUseOriginalButton.disabled) return;
    setDsExtractSelection("original");
  });
  els.dsExtractUseVocalsButton?.addEventListener("click", () => {
    if (els.dsExtractUseVocalsButton.disabled) return;
    setDsExtractSelection("vocals");
  });
  els.dsExtractConfirmButton?.addEventListener("click", () =>
    confirmExtractToDataset().catch(e => {
      state.dsExtractConfirmPending = false;
      syncDatasetExtractSummary();
      updateDatasetExtractActions();
      appendLog(els.dsExtractLogConsole, `错误：${e.message}`);
    })
  );

  // 数据集列表点击
  els.datasetCards?.addEventListener("click", (e) => {
    const card = e.target.closest("[data-dataset-id]");
    if (!card) return;
    if (state.selectedDatasetId && state.selectedDatasetId !== card.dataset.datasetId) {
      resetDatasetExtractUi();
    }
    state.selectedDatasetId = card.dataset.datasetId;
    const ds = datasetById(state.selectedDatasetId);
    state.selectedDatasetVersionId = ds?.versions?.[0]?.id || null;
    renderDatasets();
    renderDatasetDetail();
    renderTrainingConsole();
  });

  // 片段勾选
  els.datasetSegmentsList?.addEventListener("change", (e) => {
    const cb = e.target.closest("[data-segment-toggle]");
    if (!cb) return;
    toggleSegment(cb.dataset.segmentToggle, cb.checked)
      .catch(err => setText(els.datasetUploadHintText, `错误：${err.message}`));
  });

  // 版本"用于训练"按钮 → 跳到训练 Tab
  els.datasetVersionsList?.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-use-version]");
    if (!btn) return;
    state.selectedDatasetVersionId = btn.dataset.useVersion;
    renderDatasetDetail();
    renderTrainingConsole();
    setActiveView("training");
  });

  // 训练模式切换 → 续训字段显隐
  els.trainingModeSelect?.addEventListener("change", () => {
    renderTrainingConsole();
  });

  els.trainingResumeVersionSelect?.addEventListener("change", () => {
    syncResumeCheckpoints();
    const version = trainingSourceVersionDetail();
    if (version?.dataset_version_id) {
      state.selectedDatasetVersionId = version.dataset_version_id;
      if (els.trainingDatasetVersionSelect) {
        els.trainingDatasetVersionSelect.value = version.dataset_version_id;
      }
    }
    renderTrainingConsole();
  });
  els.trainingDatasetVersionSelect?.addEventListener("change", () => {
    state.selectedDatasetVersionId = els.trainingDatasetVersionSelect.value || null;
  });
  els.trainingDeviceSelect?.addEventListener("change", () => renderTrainingConsole());
  els.trainingPresetCards?.addEventListener("click", (e) => {
    const card = e.target.closest("[data-training-preset]");
    if (!card) return;
    if (card.disabled) return;
    state.trainingPresetId = card.dataset.trainingPreset;
    state.trainingUseTiny = false;
    if (els.trainingUseTinyCheckbox) {
      els.trainingUseTinyCheckbox.checked = false;
    }
    renderTrainingConsole();
  });
  els.trainingUseTinyCheckbox?.addEventListener("change", () => {
    state.trainingUseTiny = Boolean(els.trainingUseTinyCheckbox.checked);
  });

  // 开始训练
  els.startTrainingButton?.addEventListener("click", () =>
    startTraining().catch(e => appendLog(els.trainingLogConsole, `错误：${e.message}`))
  );

  // 训练完成 → 去推理
  els.jumpToInferenceButton?.addEventListener("click", () => {
    if (state.selectedInferenceProfileId && els.inferenceProfileSelect) {
      els.inferenceProfileSelect.value = state.selectedInferenceProfileId;
      syncInferenceSpeakers();
    }
    // 重置加载状态，提示用户重新加载新模型
    state.inferenceModelLoaded = false;
    updateInferenceLoadUi();
    setText(els.inferenceModelHintText, "已自动选中刚训练的版本，点击「加载模型」后即可推理。");
    setActiveView("inference");
  });

  // 模型列表点击
  els.modelCards?.addEventListener("click", (e) => {
    const card = e.target.closest("[data-model-id]");
    if (!card) return;
    state.selectedModelId = card.dataset.modelId;
    renderModels();
    renderModelDetail();
  });

  // 模型版本操作
  els.modelVersionsList?.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-model-action]");
    if (!btn) return;
    performModelAction(btn.dataset.modelAction, btn.dataset.modelId, btn.dataset.versionId)
      .catch(err => appendLog(els.trainingLogConsole, `错误：${err.message}`));
  });

  // 推理：加载模型
  els.loadInferenceModelButton?.addEventListener("click", () =>
    loadInferenceModel().catch(e => {
      setText(els.inferenceModelHintText, `错误：${e.message}`);
      state.inferenceModelLoaded = false;
      updateInferenceLoadUi();
    })
  );
  els.unloadInferenceModelButton?.addEventListener("click", () =>
    unloadInferenceModel().catch(e => setText(els.inferenceModelHintText, `错误：${e.message}`))
  );

  // 推理：选择模型时同步 speaker
  els.inferenceProfileSelect?.addEventListener("change", () => {
    state.selectedInferenceProfileId = els.inferenceProfileSelect.value;
    state.inferenceUseDiffusion = false;
    if (els.inferenceUseDiffusionCheckbox) {
      els.inferenceUseDiffusionCheckbox.checked = false;
    }
    state.inferenceModelLoaded = false; // 模型切换后需要重新加载
    updateInferenceLoadUi();
    setText(els.inferenceModelHintText, "模型版本已切换，请重新加载。");
    syncInferenceSpeakers();
  });
  els.inferenceUseDiffusionCheckbox?.addEventListener("change", () => {
    state.inferenceUseDiffusion = Boolean(els.inferenceUseDiffusionCheckbox.checked);
    state.inferenceModelLoaded = false;
    updateInferenceLoadUi();
    setText(els.inferenceModelHintText, "扩散设置已切换，请重新加载模型。");
  });

  // 推理：选择文件
  els.inferenceFileInput?.addEventListener("change", (e) => {
    const file = e.target.files?.[0] || null;
    resetPreprocessUi();
    state.inferenceFile = file;
    clearObjectUrls();
    if (file) {
      setText(els.inferenceFileNameText, `${file.name} · ${formatBytes(file.size)}`);
      const url = URL.createObjectURL(file);
      state.objectUrls.push(url);
    } else {
      setText(els.inferenceFileNameText, "尚未选择文件");
    }
    updatePreparationLockState();
    updatePreprocessActions();
  });

  els.preprocessAutoCheckbox?.addEventListener("change", () => {
    state.preprocessAutoExtract = els.preprocessAutoCheckbox.checked;
    closePreprocessRealtime();
    state.preprocessTaskId = null;
    state.preprocessTaskStatus = "idle";
    state.preprocessResult = null;
    state.preprocessSelection = state.preprocessAutoExtract ? null : (state.inferenceFile ? "original" : null);
    clearMediaElementSource(els.preprocessOriginalPlayer);
    clearMediaElementSource(els.preprocessVocalsPlayer);
    els.preprocessCompareCard?.classList.add("hidden");
    els.preprocessWarningText?.classList.add("hidden");
    setText(els.preprocessWarningText, "");
    if (els.preprocessLogConsole) {
      els.preprocessLogConsole.textContent = state.inferenceFile
        ? (state.preprocessAutoExtract ? "等待点击「提取人声」…" : "已跳过人声分离，将直接使用原始音频。")
        : "等待选择文件…";
    }
    updatePreparationLockState();
    updatePreprocessActions();
  });

  els.startPreprocessButton?.addEventListener("click", () =>
    startPreprocess().catch(error => appendLog(els.preprocessLogConsole, `错误：${error.message}`))
  );

  els.useOriginalButton?.addEventListener("click", () => {
    if (els.useOriginalButton.disabled) return;
    setVariantSelection("original");
    updatePreparationLockState();
  });

  els.useVocalsButton?.addEventListener("click", () => {
    if (els.useVocalsButton.disabled) return;
    setVariantSelection("vocals");
    updatePreparationLockState();
  });

  // 开始推理
  els.startInferenceButton?.addEventListener("click", () =>
    startInference().catch(e => appendLog(els.inferenceLogConsole, `错误：${e.message}`))
  );
}

// ── 初始化 ───────────────────────────────────────────────────────────────────
async function init() {
  // 先填充 F0 下拉（settings 里用的 select，在 overview 返回前显示占位）
  populateSelect(els.settingsDefaultF0Select, predictorOptions(), "rmvpe");
  if (els.preprocessAutoCheckbox) {
    els.preprocessAutoCheckbox.checked = false;
  }
  state.preprocessAutoExtract = false;
  if (els.dsExtractAutoSegmentCheckbox) {
    els.dsExtractAutoSegmentCheckbox.checked = true;
  }
  state.dsExtractAutoSegment = true;

  resetTrainingUi();
  resetInferenceUi();
  resetPreprocessUi();
  resetDatasetExtractUi();
  bindEvents();

  try {
    await refreshOverview();
  } catch (e) {
    setText(els.settingsHintText, `初始化失败：${e.message}`);
    appendLog(els.trainingLogConsole, `初始化失败：${e.message}`);
  }
}

window.addEventListener("beforeunload", () => {
  state.isUnloading = true;
  closeTrainingRealtime();
  closeInferenceRealtime();
  closePreprocessRealtime();
  closeDatasetExtractRealtime();
  clearObjectUrls();
});

window.addEventListener("DOMContentLoaded", init);
