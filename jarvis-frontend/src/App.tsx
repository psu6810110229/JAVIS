import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import PowerController from "./PowerController";

type SocketStatus = "disconnected" | "connecting" | "connected";
type PowerMode = "eco" | "performance";

type Envelope = {
  type: string;
  payload: Record<string, unknown>;
  timestamp?: string;
  session_id?: string;
};

type ConfirmationRequest = {
  tool_name: string;
  human_label: string;
  risk_level: "medium" | "high";
  message: string;
  timeout_seconds: number;
};

type ChatMessage = {
  id: string;
  role: "system" | "user" | "assistant";
  label: string;
  content: string;
  toolOutcome?: ToolOutcomeSummary;
};

type ToolOutcomeSummary = {
  tool_name: string;
  status: string;
  verified: boolean | null;
  evidence: string | null;
};

type AssistantAudioPayload = {
  audio_base64: string;
  mime_type: string;
  voice: string;
};

type SystemTelemetry = {
  gpuLoad: number;
  ramAvailableGb: number;
  diskIoMBps: number;
  telemetrySource: string;
};

type SystemCapabilities = {
  numBatchSupported: boolean;
  numBatchWarning: string;
  hostOptimizerStatus: string;
};

const DEFAULT_WS_URL = "ws://127.0.0.1:8000/ws";
const DEFAULT_API_URL = "http://127.0.0.1:8000";
const POWER_MODE_STORAGE_KEY = "jarvis.power.mode";
const VOICE_AUTO_SPEAK_STORAGE_KEY = "jarvis.voice.auto_speak";
const PTT_HOTKEY = "Alt";
const TURBO_MODE_THRESHOLD = 95;
const EOF_RETRY_DELAY_MS = 500;
const EOF_RETRY_NUM_CTX = 512;

function readPersistedPowerMode(): PowerMode {
  const persisted = window.localStorage.getItem(POWER_MODE_STORAGE_KEY);
  return persisted === "performance" ? "performance" : "eco";
}

function getModeDisplayLabel(mode: PowerMode): string {
  return mode === "performance" ? "Deep" : "Quick";
}

function readPersistedVoiceAutoSpeak(): boolean {
  const persisted = window.localStorage.getItem(VOICE_AUTO_SPEAK_STORAGE_KEY);
  if (persisted === null) {
    return false;
  }

  return persisted !== "off";
}

function base64ToBlob(base64: string, mimeType: string): Blob {
  const binaryString = window.atob(base64);
  const bytes = new Uint8Array(binaryString.length);

  for (let index = 0; index < binaryString.length; index += 1) {
    bytes[index] = binaryString.charCodeAt(index);
  }

  return new Blob([bytes], { type: mimeType });
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result;
      if (typeof result !== "string") {
        reject(new Error("Recorded audio could not be converted to Base64."));
        return;
      }

      const base64Payload = result.split(",")[1];
      if (!base64Payload) {
        reject(new Error("Recorded audio produced an invalid Base64 payload."));
        return;
      }

      resolve(base64Payload);
    };
    reader.onerror = () => {
      reject(new Error("Recorded audio could not be read by FileReader."));
    };
    reader.readAsDataURL(blob);
  });
}

function parseToolOutcome(value: unknown): ToolOutcomeSummary | undefined {
  if (!value || typeof value !== "object") {
    return undefined;
  }

  const payload = value as Record<string, unknown>;
  const toolName = typeof payload.tool_name === "string" ? payload.tool_name : "";
  if (!toolName) {
    return undefined;
  }

  const status = typeof payload.status === "string" ? payload.status : "unknown";
  const verifiedRaw = payload.verified;
  const verified = typeof verifiedRaw === "boolean" ? verifiedRaw : null;
  const evidence = typeof payload.evidence === "string" ? payload.evidence : null;

  return {
    tool_name: toolName,
    status,
    verified,
    evidence
  };
}

function App() {
  const [socketStatus, setSocketStatus] = useState<SocketStatus>("disconnected");
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "system",
      label: "System",
      content: "Jarvis shell is ready. Connect to the backend to start a session."
    }
  ]);
  const [inputValue, setInputValue] = useState<string>("");
  const [sessionId, setSessionId] = useState<string>("");
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [isPlayingAudio, setIsPlayingAudio] = useState<boolean>(false);
  const [micError, setMicError] = useState<string>("");
  const [audioUiError, setAudioUiError] = useState<string>("");
  const [sttUiError, setSttUiError] = useState<string>("");
  const [voiceAutoSpeak, setVoiceAutoSpeak] = useState<boolean>(() => readPersistedVoiceAutoSpeak());
  const [currentMode, setCurrentMode] = useState<PowerMode>(() => readPersistedPowerMode());
  const [activeModel, setActiveModel] = useState<string>("");
  const [isSwitchingMode, setIsSwitchingMode] = useState<boolean>(false);
  const [statusLoadError, setStatusLoadError] = useState<string>("");
  const [prewarmWarning, setPrewarmWarning] = useState<string>("");
  const [pendingAssistantResponse, setPendingAssistantResponse] = useState<boolean>(false);
  const [confirmationRequest, setConfirmationRequest] = useState<ConfirmationRequest | null>(null);
  const [confirmCountdown, setConfirmCountdown] = useState<number>(30);
  const confirmCountdownRef = useRef<number | null>(null);
  const [awaitingAssistantAudio, setAwaitingAssistantAudio] = useState<boolean>(false);
  const [isTextStreaming, setIsTextStreaming] = useState<boolean>(false);
  const [showVoiceControls, setShowVoiceControls] = useState<boolean>(false);
  const [showModelContext, setShowModelContext] = useState<boolean>(false);
  const [transportNotice, setTransportNotice] = useState<string>("");
  const [manualListenMessageId, setManualListenMessageId] = useState<string>("");
  const [telemetry, setTelemetry] = useState<SystemTelemetry>({
    gpuLoad: 0,
    ramAvailableGb: 0,
    diskIoMBps: 0,
    telemetrySource: "unavailable"
  });
  const [capabilities, setCapabilities] = useState<SystemCapabilities>({
    numBatchSupported: true,
    numBatchWarning: "",
    hostOptimizerStatus: "unknown"
  });
  const socketRef = useRef<WebSocket | null>(null);
  const chatAbortControllerRef = useRef<AbortController | null>(null);
  const activeHttpAssistantMessageIdRef = useRef<string | null>(null);
  const messagesViewportRef = useRef<HTMLDivElement | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const manualDisconnectRef = useRef<boolean>(false);
  const pttHeldRef = useRef<boolean>(false);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<BlobPart[]>([]);
  const shouldSendRecordingRef = useRef<boolean>(true);
  const activeAudioRef = useRef<HTMLAudioElement | null>(null);
  const activeAudioUrlRef = useRef<string>("");
  const activeAudioTokenRef = useRef<number>(0);
  const audioLoadConfirmedRef = useRef<boolean>(false);
  const audioPlaybackErroredRef = useRef<boolean>(false);
  const audioErrorTimeoutRef = useRef<number | null>(null);
  const audioUiHideTimeoutRef = useRef<number | null>(null);
  const sttUiHideTimeoutRef = useRef<number | null>(null);
  const modelContextTimeoutRef = useRef<number | null>(null);
  const activeStreamIdRef = useRef<string | null>(null);
  const activeStreamMessageIdRef = useRef<string | null>(null);
  const autoSpeakQueueRef = useRef<AssistantAudioPayload[]>([]);
  const autoSpeakPlayingRef = useRef<boolean>(false);
  const wsUrl = import.meta.env.VITE_JARVIS_WS_URL ?? DEFAULT_WS_URL;
  const apiBaseUrl = import.meta.env.VITE_JARVIS_API_URL ?? DEFAULT_API_URL;
  const mediaRecorderSupported =
    typeof window !== "undefined" &&
    typeof navigator !== "undefined" &&
    typeof navigator.mediaDevices !== "undefined" &&
    typeof navigator.mediaDevices.getUserMedia === "function" &&
    typeof MediaRecorder !== "undefined";
  const isRequestActive =
    pendingAssistantResponse || isTextStreaming || awaitingAssistantAudio || isPlayingAudio;

  async function hydrateStatus(silentFailure: boolean = false): Promise<void> {
    try {
      const response = await fetch(`${apiBaseUrl}/v1/system/status`);
      const payload = (await response.json()) as Record<string, unknown>;
      if (!response.ok) {
        throw new Error(
          typeof payload.detail === "string"
            ? payload.detail
            : "Jarvis could not load system status."
        );
      }

      const serverMode =
        payload.active_mode === "performance" || payload.active_mode === "eco"
          ? (payload.active_mode as PowerMode)
          : "eco";
      const serverModel = typeof payload.active_model === "string" ? payload.active_model : "";
      setCurrentMode(serverMode);
      setActiveModel(serverModel);
      window.localStorage.setItem(POWER_MODE_STORAGE_KEY, serverMode);

      const systemLoad =
        typeof payload.system_load === "object" && payload.system_load !== null
          ? (payload.system_load as Record<string, unknown>)
          : {};

      const gpuLoad = typeof systemLoad.GPU_Load === "number" ? systemLoad.GPU_Load : 0;
      const ramAvailableGb = typeof systemLoad.RAM_Available === "number" ? systemLoad.RAM_Available : 0;
      const diskIoMBps =
        typeof systemLoad.Disk_IO_MBps === "number"
          ? systemLoad.Disk_IO_MBps
          : typeof systemLoad.Disk_IO === "number"
            ? systemLoad.Disk_IO
            : 0;
      const telemetrySource =
        typeof systemLoad.telemetry_source === "string" ? systemLoad.telemetry_source : "unavailable";

      const capabilityPayload =
        typeof payload.capabilities === "object" && payload.capabilities !== null
          ? (payload.capabilities as Record<string, unknown>)
          : {};
      const hostOptimizerPayload =
        typeof capabilityPayload.host_optimizer === "object" && capabilityPayload.host_optimizer !== null
          ? (capabilityPayload.host_optimizer as Record<string, unknown>)
          : {};

      setCapabilities({
        numBatchSupported:
          typeof capabilityPayload.num_batch_supported === "boolean"
            ? capabilityPayload.num_batch_supported
            : true,
        numBatchWarning:
          typeof capabilityPayload.num_batch_warning === "string" ? capabilityPayload.num_batch_warning : "",
        hostOptimizerStatus:
          typeof hostOptimizerPayload.status === "string" ? hostOptimizerPayload.status : "unknown"
      });

      setTelemetry({
        gpuLoad,
        ramAvailableGb,
        diskIoMBps,
        telemetrySource
      });
      setStatusLoadError("");
    } catch (error: unknown) {
      if (!silentFailure) {
        setStatusLoadError(
          error instanceof Error ? error.message : "Jarvis could not load the current model mode status."
        );
      }
    }
  }

  useEffect(() => {
    return () => {
      chatAbortControllerRef.current?.abort();
      chatAbortControllerRef.current = null;
      autoSpeakQueueRef.current = [];
      autoSpeakPlayingRef.current = false;
      stopRecording(true);
      clearAudioUiHideTimeout();
      clearSttUiHideTimeout();
      clearModelContextTimeout();
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      cleanupAudioPlayback();
      manualDisconnectRef.current = true;
      socketRef.current?.close();
    };
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== PTT_HOTKEY || event.repeat || pttHeldRef.current) {
        return;
      }

      pttHeldRef.current = true;

      if (socketStatus !== "connected") {
        setMicError("Jarvis voice input is unavailable while disconnected.");
        return;
      }

      if (!mediaRecorderSupported) {
        setMicError("This runtime does not support keyboard push-to-talk recording.");
        return;
      }

      void startRecording();
    };

    const onKeyUp = (event: KeyboardEvent) => {
      if (event.key !== PTT_HOTKEY) {
        return;
      }

      pttHeldRef.current = false;
      stopRecording();
    };

    const onWindowBlur = () => {
      if (!pttHeldRef.current) {
        return;
      }

      pttHeldRef.current = false;
      stopRecording();
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onWindowBlur);

    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", onWindowBlur);
    };
  }, [socketStatus, mediaRecorderSupported, isRequestActive]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      connect();
    }, 80);

    return () => {
      window.clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    if (!messagesViewportRef.current) {
      return;
    }

    messagesViewportRef.current.scrollTo({
      top: messagesViewportRef.current.scrollHeight,
      behavior: "smooth"
    });
  }, [messages, pendingAssistantResponse]);

  useEffect(() => {
    let isDisposed = false;

    const runHydrate = async () => {
      if (isDisposed) {
        return;
      }
      await hydrateStatus(false);
    };

    void runHydrate();
    const intervalId = window.setInterval(() => {
      void hydrateStatus(true);
    }, 2000);

    return () => {
      isDisposed = true;
      window.clearInterval(intervalId);
    };
  }, [apiBaseUrl]);

  function pushMessage(message: ChatMessage): void {
    setMessages((current) => [...current, message]);
  }

  function isToolCallPayloadText(content: string): boolean {
    const trimmed = content.trim();
    if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) {
      return false;
    }

    try {
      const parsed = JSON.parse(trimmed) as Record<string, unknown>;
      const declaredType = typeof parsed.t === "string" ? parsed.t : parsed.type;
      const toolName =
        typeof parsed.n === "string"
          ? parsed.n
          : typeof parsed.tool_name === "string"
            ? parsed.tool_name
            : typeof parsed.name === "string"
              ? parsed.name
              : "";
      return (declaredType === "tool" || declaredType === undefined) && toolName.length > 0;
    } catch {
      return false;
    }
  }

  function normalizeAssistantContent(content: string): string {
    if (isToolCallPayloadText(content)) {
      return "Processing command...";
    }
    return content;
  }

  function updateMessageContent(messageId: string, content: string): void {
    setMessages((current) =>
      current.map((message) =>
        message.id === messageId ? { ...message, content: normalizeAssistantContent(content) } : message
      )
    );
  }

  function updateMessageToolOutcome(messageId: string, toolOutcome: ToolOutcomeSummary | undefined): void {
    setMessages((current) =>
      current.map((message) => (message.id === messageId ? { ...message, toolOutcome } : message))
    );
  }

  function appendMessageContent(messageId: string, delta: string): void {
    setMessages((current) =>
      current.map((message) =>
        message.id === messageId
          ? { ...message, content: normalizeAssistantContent(`${message.content}${delta}`) }
          : message
      )
    );
  }

  function persistPowerMode(mode: PowerMode): void {
    window.localStorage.setItem(POWER_MODE_STORAGE_KEY, mode);
  }

  function persistVoiceAutoSpeak(enabled: boolean): void {
    window.localStorage.setItem(VOICE_AUTO_SPEAK_STORAGE_KEY, enabled ? "on" : "off");
  }

  function clearModelContextTimeout(): void {
    if (modelContextTimeoutRef.current !== null) {
      window.clearTimeout(modelContextTimeoutRef.current);
      modelContextTimeoutRef.current = null;
    }
  }

  async function handleSwitchMode(nextMode: PowerMode): Promise<void> {
    if (isSwitchingMode || nextMode === currentMode) {
      return;
    }

    setShowModelContext(true);
    setIsSwitchingMode(true);
    setStatusLoadError("");
    setPrewarmWarning("");
    setMicError("");

    try {
      const response = await fetch(`${apiBaseUrl}/v1/system/model`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ mode: nextMode })
      });

      const payload = (await response.json()) as Record<string, unknown>;
      if (!response.ok) {
        throw new Error(
          typeof payload.detail === "string" ? payload.detail : "Jarvis could not switch model mode."
        );
      }

      const activeMode =
        payload.active_mode === "performance" || payload.active_mode === "eco"
          ? (payload.active_mode as PowerMode)
          : nextMode;
      const activeModelName = typeof payload.active_model === "string" ? payload.active_model : "";
      const warning = typeof payload.prewarm_warning === "string" ? payload.prewarm_warning : "";

      setCurrentMode(activeMode);
      setActiveModel(activeModelName);
      persistPowerMode(activeMode);
      setPrewarmWarning(warning);

      pushMessage({
        id: `mode-switch-${Date.now()}`,
        role: "system",
        label: "Model",
        content:
          `Switched to ${getModeDisplayLabel(activeMode)} mode.`
      });
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : "Jarvis could not switch model mode right now.";
      setStatusLoadError(message);
      pushMessage({
        id: `mode-switch-error-${Date.now()}`,
        role: "system",
        label: "Model",
        content: message
      });
    } finally {
      setIsSwitchingMode(false);
      setTransportNotice("");
      clearModelContextTimeout();
      modelContextTimeoutRef.current = window.setTimeout(() => {
        setShowModelContext(false);
        modelContextTimeoutRef.current = null;
      }, 950);
    }
  }

  function cleanupMediaStream(): void {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }

  function clearAudioUiHideTimeout(): void {
    if (audioUiHideTimeoutRef.current !== null) {
      window.clearTimeout(audioUiHideTimeoutRef.current);
      audioUiHideTimeoutRef.current = null;
    }
  }

  function clearSttUiHideTimeout(): void {
    if (sttUiHideTimeoutRef.current !== null) {
      window.clearTimeout(sttUiHideTimeoutRef.current);
      sttUiHideTimeoutRef.current = null;
    }
  }

  function showAudioUiError(content: string): void {
    clearAudioUiHideTimeout();
    setAudioUiError(content);
    audioUiHideTimeoutRef.current = window.setTimeout(() => {
      setAudioUiError("");
      audioUiHideTimeoutRef.current = null;
    }, 3000);
  }

  function clearAudioUiError(): void {
    clearAudioUiHideTimeout();
    setAudioUiError("");
  }

  function showSttUiError(content: string): void {
    clearSttUiHideTimeout();
    setSttUiError(content);
    sttUiHideTimeoutRef.current = window.setTimeout(() => {
      setSttUiError("");
      sttUiHideTimeoutRef.current = null;
    }, 3000);
  }

  function clearSttUiError(): void {
    clearSttUiHideTimeout();
    setSttUiError("");
  }

  function clearAudioErrorTimeout(): void {
    if (audioErrorTimeoutRef.current !== null) {
      window.clearTimeout(audioErrorTimeoutRef.current);
      audioErrorTimeoutRef.current = null;
    }
  }

  function scheduleAudioError(content: string, token: number, delayMs: number = 2000): void {
    clearAudioErrorTimeout();
    audioErrorTimeoutRef.current = window.setTimeout(() => {
      if (token !== activeAudioTokenRef.current || audioLoadConfirmedRef.current) {
        return;
      }

      audioPlaybackErroredRef.current = true;
      showAudioUiError(content);
      cleanupAudioPlayback();
    }, delayMs);
  }

  function cleanupAudioPlayback(): void {
    clearAudioErrorTimeout();
    activeAudioTokenRef.current += 1;
    audioLoadConfirmedRef.current = false;
    audioPlaybackErroredRef.current = false;

    if (activeAudioRef.current) {
      activeAudioRef.current.onended = null;
      activeAudioRef.current.onerror = null;
      activeAudioRef.current.oncanplaythrough = null;
      activeAudioRef.current.onloadeddata = null;
      activeAudioRef.current.pause();
      activeAudioRef.current.src = "";
      activeAudioRef.current = null;
    }

    if (activeAudioUrlRef.current) {
      URL.revokeObjectURL(activeAudioUrlRef.current);
      activeAudioUrlRef.current = "";
    }

    setManualListenMessageId("");
    setIsPlayingAudio(false);
  }

  async function playAssistantAudio(
    payload: AssistantAudioPayload,
    onPlaybackComplete?: () => void
  ): Promise<void> {
    cleanupAudioPlayback();

    try {
      const playbackToken = activeAudioTokenRef.current;
      const audioBlob = base64ToBlob(payload.audio_base64, payload.mime_type);
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);

      activeAudioRef.current = audio;
      activeAudioUrlRef.current = audioUrl;
      audioLoadConfirmedRef.current = false;
      audioPlaybackErroredRef.current = false;
      setIsPlayingAudio(true);

      const markLoaded = () => {
        if (playbackToken !== activeAudioTokenRef.current) {
          return;
        }
        audioLoadConfirmedRef.current = true;
        audioPlaybackErroredRef.current = false;
        clearAudioErrorTimeout();
        clearAudioUiError();
      };

      audio.onloadeddata = markLoaded;
      audio.oncanplaythrough = markLoaded;
      audio.onplay = markLoaded;

      audio.onended = () => {
        if (playbackToken !== activeAudioTokenRef.current) {
          return;
        }
        cleanupAudioPlayback();
        onPlaybackComplete?.();
      };
      audio.onerror = () => {
        if (playbackToken !== activeAudioTokenRef.current || audioLoadConfirmedRef.current) {
          return;
        }

        scheduleAudioError("Jarvis could not play the assistant audio reply.", playbackToken);
        onPlaybackComplete?.();
      };

      await audio.play();
      markLoaded();
    } catch (error: unknown) {
      const isAbortError = error instanceof DOMException && error.name === "AbortError";
      if (isAbortError) {
        onPlaybackComplete?.();
        return;
      }

      cleanupAudioPlayback();
      showAudioUiError(
        error instanceof Error ? error.message : "Jarvis encountered an unexpected audio playback error."
      );
      onPlaybackComplete?.();
    }
  }

  function clearAutoSpeakQueue(): void {
    autoSpeakQueueRef.current = [];
    autoSpeakPlayingRef.current = false;
    setAwaitingAssistantAudio(false);
  }

  function playNextAutoSpeakAudio(): void {
    if (autoSpeakPlayingRef.current) {
      return;
    }

    const nextAudio = autoSpeakQueueRef.current.shift();
    if (!nextAudio) {
      setAwaitingAssistantAudio(false);
      return;
    }

    autoSpeakPlayingRef.current = true;
    setAwaitingAssistantAudio(true);

    void playAssistantAudio(nextAudio, () => {
      autoSpeakPlayingRef.current = false;
      if (autoSpeakQueueRef.current.length === 0) {
        setAwaitingAssistantAudio(false);
        return;
      }
      playNextAutoSpeakAudio();
    });
  }

  function enqueueAutoSpeakAudio(payload: AssistantAudioPayload): void {
    autoSpeakQueueRef.current.push(payload);
    playNextAutoSpeakAudio();
  }

  async function handleManualListen(messageId: string, text: string): Promise<void> {
    const content = text.trim();
    if (!content) {
      return;
    }

    clearAutoSpeakQueue();
    cleanupAudioPlayback();
    setManualListenMessageId(messageId);

    try {
      const response = await fetch(`${apiBaseUrl}/v1/synthesize`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: content })
      });

      const payload = (await response.json()) as Record<string, unknown>;
      if (!response.ok) {
        throw new Error(
          typeof payload.detail === "string"
            ? payload.detail
            : "Jarvis could not synthesize this message right now."
        );
      }

      if (
        typeof payload.audio_base64 !== "string" ||
        typeof payload.mime_type !== "string" ||
        typeof payload.voice !== "string"
      ) {
        throw new Error("Jarvis returned an invalid manual synthesis payload.");
      }

      await playAssistantAudio(
        {
          audio_base64: payload.audio_base64,
          mime_type: payload.mime_type,
          voice: payload.voice
        },
        () => {
          setManualListenMessageId("");
        }
      );
    } catch (error: unknown) {
      setManualListenMessageId("");
      showAudioUiError(
        error instanceof Error ? error.message : "Jarvis could not play the selected message audio."
      );
    }
  }

  async function streamChatOverHttp(
    userText: string,
    assistantMessageId: string,
    attempt: number = 0,
    numCtxOverride?: number
  ): Promise<void> {
    const requestController = new AbortController();
    chatAbortControllerRef.current = requestController;
    activeHttpAssistantMessageIdRef.current = assistantMessageId;

    setPendingAssistantResponse(true);
    setIsTextStreaming(false);
    setAwaitingAssistantAudio(false);
    setTransportNotice(`Processing in ${getModeDisplayLabel(currentMode)} mode...`);

    try {
      const response = await fetch(`${apiBaseUrl}/v1/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          message: userText,
          auto_speak: voiceAutoSpeak,
          session_id: sessionId || undefined,
          num_ctx: numCtxOverride
        }),
        signal: requestController.signal
      });

      if (!response.ok) {
        const payload = (await response.json()) as Record<string, unknown>;
        throw new Error(
          typeof payload.detail === "string"
            ? payload.detail
            : "Jarvis could not process this request right now."
        );
      }

      if (!response.body) {
        throw new Error("Jarvis returned an empty stream response.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let streamErrorMessage = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          const trimmedLine = line.trim();
          if (!trimmedLine) {
            continue;
          }

          let event: Record<string, unknown>;
          try {
            event = JSON.parse(trimmedLine) as Record<string, unknown>;
          } catch {
            continue;
          }

          if (typeof event.session_id === "string" && event.session_id.length > 0) {
            setSessionId(event.session_id);
          }

          const eventType = typeof event.type === "string" ? event.type : "";
          if (eventType === "text.chunk") {
            const delta = typeof event.delta === "string" ? event.delta : "";
            if (delta) {
              setPendingAssistantResponse(false);
              setIsTextStreaming(true);
              appendMessageContent(assistantMessageId, delta);
            }
            continue;
          }

          if (eventType === "audio.ready" && voiceAutoSpeak) {
            if (
              typeof event.audio_base64 === "string" &&
              typeof event.mime_type === "string" &&
              typeof event.voice === "string"
            ) {
              enqueueAutoSpeakAudio({
                audio_base64: event.audio_base64,
                mime_type: event.mime_type,
                voice: event.voice
              });
            }
            continue;
          }

          if (eventType === "final") {
            const finalText = typeof event.text === "string" ? event.text.trim() : "";
            const toolOutcome = parseToolOutcome(event.last_tool_outcome);
            if (finalText.length > 0) {
              updateMessageContent(assistantMessageId, finalText);
            }
            updateMessageToolOutcome(assistantMessageId, toolOutcome);
            setPendingAssistantResponse(false);
            setIsTextStreaming(false);
            if (!voiceAutoSpeak || (autoSpeakQueueRef.current.length === 0 && !autoSpeakPlayingRef.current)) {
              setAwaitingAssistantAudio(false);
            }
            continue;
          }

          if (eventType === "error") {
            const stage = typeof event.stage === "string" ? event.stage : "";
            const message =
              typeof event.message === "string"
                ? event.message
                : "Jarvis encountered an unexpected streaming error.";

            if (stage === "tts") {
              pushMessage({
                id: `tts-warning-${Date.now()}`,
                role: "system",
                label: "Voice",
                content: message
              });
              continue;
            }

            streamErrorMessage = message;
            break;
          }
        }
      }

      if (streamErrorMessage) {
        const normalized = streamErrorMessage.toLowerCase();
        const isUnexpectedEof =
          normalized.includes("unexpected eof") ||
          normalized.includes("incomplete") ||
          normalized.includes("connection reset");

        if (isUnexpectedEof && attempt < 1) {
          setTransportNotice("Stream interrupted. Retrying with low context...");
          await new Promise((resolve) => window.setTimeout(resolve, EOF_RETRY_DELAY_MS));
          await streamChatOverHttp(userText, assistantMessageId, attempt + 1, EOF_RETRY_NUM_CTX);
          return;
        }

        pushMessage({
          id: `stream-error-${Date.now()}`,
          role: "system",
          label: "Error",
          content: streamErrorMessage
        });
      }
    } catch (error: unknown) {
      const isAbortError = error instanceof DOMException && error.name === "AbortError";
      if (!isAbortError) {
        pushMessage({
          id: `http-chat-error-${Date.now()}`,
          role: "system",
          label: "Error",
          content:
            error instanceof Error
              ? error.message
              : "Jarvis could not complete the HTTP chat request."
        });
      }
    } finally {
      if (chatAbortControllerRef.current === requestController) {
        chatAbortControllerRef.current = null;
      }
      activeHttpAssistantMessageIdRef.current = null;
      setPendingAssistantResponse(false);
      setIsTextStreaming(false);
      if (autoSpeakQueueRef.current.length === 0 && !autoSpeakPlayingRef.current) {
        setAwaitingAssistantAudio(false);
      }
      setTransportNotice("");
    }
  }

  function connect(): void {
    if (socketRef.current?.readyState === WebSocket.OPEN || socketStatus === "connecting") {
      return;
    }

    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }

    manualDisconnectRef.current = false;
    setTransportNotice("Connecting...");
    setSocketStatus("connecting");
    const socket = new WebSocket(wsUrl);
    socketRef.current = socket;

    socket.onopen = () => {
      setSocketStatus("connected");
      setTransportNotice("");
      socket.send(
        JSON.stringify({
          type: "session.start",
          payload: {
            client_name: "jarvis-frontend"
          }
        } satisfies Envelope)
      );
    };

    socket.onmessage = (event: MessageEvent<string>) => {
      let data: Envelope;
      try {
        data = JSON.parse(event.data) as Envelope;
      } catch (error: unknown) {
        pushMessage({
          id: `socket-parse-${Date.now()}`,
          role: "system",
          label: "Error",
          content:
            error instanceof SyntaxError
              ? "Jarvis received a malformed backend message."
              : "Jarvis received an unexpected socket parsing error."
        });
        return;
      }

      if (data.session_id) {
        setSessionId(data.session_id);
      }

      if (data.type === "text.received" || data.type === "audio.ack") {
        return;
      }

      if (data.type === "confirmation_required") {
        const p = data.payload as Partial<ConfirmationRequest>;
        const timeout = typeof p.timeout_seconds === "number" ? p.timeout_seconds : 30;
        setConfirmationRequest({
          tool_name: typeof p.tool_name === "string" ? p.tool_name : "",
          human_label: typeof p.human_label === "string" ? p.human_label : p.tool_name ?? "",
          risk_level: p.risk_level === "high" ? "high" : "medium",
          message: typeof p.message === "string" ? p.message : "Confirm this action?",
          timeout_seconds: timeout,
        });
        setConfirmCountdown(timeout);
        if (confirmCountdownRef.current !== null) {
          window.clearInterval(confirmCountdownRef.current);
        }
        confirmCountdownRef.current = window.setInterval(() => {
          setConfirmCountdown((prev) => {
            if (prev <= 1) {
              if (confirmCountdownRef.current !== null) {
                window.clearInterval(confirmCountdownRef.current);
                confirmCountdownRef.current = null;
              }
              setConfirmationRequest(null);
              return 0;
            }
            return prev - 1;
          });
        }, 1000);
        return;
      }

      if (data.type === "tool.confirm.ack" || data.type === "tool.deny.ack") {
        return;
      }

      if (data.type === "speech.transcript") {
        const transcriptText = typeof data.payload.text === "string" ? data.payload.text : "";
        if (transcriptText) {
          clearSttUiError();
          pushMessage({
            id: `transcript-${Date.now()}`,
            role: "user",
            label: "Voice",
            content: transcriptText
          });
        }
        return;
      }

      if (data.type === "assistant_audio") {
        const payload = data.payload as Partial<AssistantAudioPayload>;
        if (
          typeof payload.audio_base64 === "string" &&
          typeof payload.mime_type === "string" &&
          typeof payload.voice === "string"
        ) {
          setAwaitingAssistantAudio(false);
          clearSttUiError();
          void playAssistantAudio(payload as AssistantAudioPayload);
        } else {
          setAwaitingAssistantAudio(false);
          showAudioUiError("Jarvis returned an invalid assistant audio payload.");
        }
        return;
      }

      if (data.type === "text.stream.start") {
        const streamId = typeof data.payload.stream_id === "string" ? data.payload.stream_id : "";
        if (!streamId) {
          return;
        }

        const messageId = `assistant-stream-${streamId}`;
        activeStreamIdRef.current = streamId;
        activeStreamMessageIdRef.current = messageId;
        setIsTextStreaming(true);
        setPendingAssistantResponse(false);
        setAwaitingAssistantAudio(false);
        setTransportNotice("");
        clearSttUiError();

        pushMessage({
          id: messageId,
          role: "assistant",
          label: "Jarvis",
          content: ""
        });
        return;
      }

      if (data.type === "text.stream.delta") {
        const streamId = typeof data.payload.stream_id === "string" ? data.payload.stream_id : "";
        const delta = typeof data.payload.delta === "string" ? data.payload.delta : "";
        const activeStreamId = activeStreamIdRef.current;
        const activeMessageId = activeStreamMessageIdRef.current;

        if (!streamId || !delta || !activeStreamId || streamId !== activeStreamId || !activeMessageId) {
          return;
        }

        appendMessageContent(activeMessageId, delta);
        return;
      }

      if (data.type === "text.stream.end") {
        const streamId = typeof data.payload.stream_id === "string" ? data.payload.stream_id : "";
        const finalText = typeof data.payload.text === "string" ? data.payload.text : "";
        const activeStreamId = activeStreamIdRef.current;
        const activeMessageId = activeStreamMessageIdRef.current;

        if (!streamId || !activeStreamId || streamId !== activeStreamId || !activeMessageId) {
          return;
        }

        if (finalText.trim().length > 0) {
          updateMessageContent(activeMessageId, finalText.trim());
        }

        setIsTextStreaming(false);
        setAwaitingAssistantAudio(true);
        // Keep stream refs until text.output arrives so we can finalize without duplicating the message.
        return;
      }

      if (data.type === "text.output") {
        const textOut = typeof data.payload.text === "string" ? data.payload.text.trim() : "";
        if (textOut.length > 0) {
          clearSttUiError();
        }

        const activeStreamMessageId = activeStreamMessageIdRef.current;
        if (activeStreamMessageId) {
          if (textOut.length > 0) {
            updateMessageContent(activeStreamMessageId, textOut);
          }

          activeStreamIdRef.current = null;
          activeStreamMessageIdRef.current = null;
          setIsTextStreaming(false);
          setPendingAssistantResponse(false);
          setAwaitingAssistantAudio(true);
          setTransportNotice("");
          return;
        }

        setPendingAssistantResponse(false);
        setAwaitingAssistantAudio(true);
        setTransportNotice("");
        if (textOut.length > 0) {
          pushMessage({
            id: `text-output-${Date.now()}`,
            role: "assistant",
            label: "Jarvis",
            content: textOut
          });
        }
        return;
      }

      if (data.type === "error") {
        activeStreamIdRef.current = null;
        activeStreamMessageIdRef.current = null;
        setIsTextStreaming(false);
        setPendingAssistantResponse(false);
        setAwaitingAssistantAudio(false);
        setTransportNotice("");
        const errorMessage = typeof data.payload.message === "string" ? data.payload.message : "";
        const normalized = errorMessage.toLowerCase();
        const isSttError =
          normalized.includes("could not understand") ||
          normalized.includes("stt") ||
          normalized.includes("spoken text") ||
          normalized.includes("recorded thai speech");

        if (isSttError) {
          showSttUiError(errorMessage || "Jarvis could not understand the recorded speech.");
          return;
        }
      }

      const payloadText =
        typeof data.payload.text === "string"
          ? data.payload.text
          : typeof data.payload.message === "string"
            ? data.payload.message
            : JSON.stringify(data.payload, null, 2);

      pushMessage({
        id: `${data.type}-${Date.now()}`,
        role: data.type === "error" ? "system" : "assistant",
        label: data.type,
        content: payloadText
      });
    };

    socket.onerror = () => {
      setTransportNotice("Connection unstable. Retrying...");
      pushMessage({
        id: `socket-error-${Date.now()}`,
        role: "system",
        label: "Error",
        content: "The Jarvis backend connection failed."
      });
    };

    socket.onclose = () => {
      stopRecording(true);
      clearAutoSpeakQueue();
      cleanupAudioPlayback();
      activeStreamIdRef.current = null;
      activeStreamMessageIdRef.current = null;
      setIsTextStreaming(false);
      setPendingAssistantResponse(false);
      setAwaitingAssistantAudio(false);
      setSocketStatus("disconnected");
      socketRef.current = null;

      if (!manualDisconnectRef.current) {
        setTransportNotice("Disconnected. Reconnecting...");
        reconnectTimerRef.current = window.setTimeout(() => {
          reconnectTimerRef.current = null;
          connect();
        }, 1200);
      }
    };
  }

  function disconnect(): void {
    stopRecording(true);
    clearAutoSpeakQueue();
    cleanupAudioPlayback();
    manualDisconnectRef.current = true;
    setTransportNotice("Disconnected");

    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }

    const socket = socketRef.current;
    if (!socket) {
      return;
    }

    if (socket.readyState === WebSocket.OPEN) {
      socket.send(
        JSON.stringify({
          type: "session.end",
          payload: {},
          session_id: sessionId
        } satisfies Envelope)
      );
    }

    socket.close();
    socketRef.current = null;
    setSocketStatus("disconnected");
  }

  async function sendRecordedAudio(audioBlob: Blob): Promise<void> {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      pushMessage({
        id: `audio-socket-missing-${Date.now()}`,
        role: "system",
        label: "Mic",
        content: "Connect to Jarvis before using keyboard push-to-talk."
      });
      setMicError("Keyboard push-to-talk failed because Jarvis is disconnected.");
      return;
    }

    if (isRequestActive) {
      setMicError("Wait until Jarvis finishes the current response before sending voice input.");
      return;
    }

    try {
      const audioBase64 = await blobToBase64(audioBlob);
      setPendingAssistantResponse(true);
      socket.send(
        JSON.stringify({
          type: "audio.chunk",
          payload: {
            data: audioBase64,
            mime_type: audioBlob.type || "audio/webm",
            is_final: true
          },
          session_id: sessionId
        } satisfies Envelope)
      );
    } catch (error: unknown) {
      setPendingAssistantResponse(false);
      setAwaitingAssistantAudio(false);
      pushMessage({
        id: `audio-send-error-${Date.now()}`,
        role: "system",
        label: "Mic",
        content:
          error instanceof Error
            ? error.message
            : "Jarvis could not serialize the microphone audio."
      });
    }
  }

  async function startRecording(): Promise<void> {
    if (isRecording || socketStatus !== "connected") {
      return;
    }

    if (isRequestActive) {
      setMicError("Wait until Jarvis finishes the current response before recording.");
      return;
    }

    if (!mediaRecorderSupported) {
      setMicError("This browser runtime does not support MediaRecorder microphone capture.");
      return;
    }

    setMicError("");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const preferredMimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/webm")
          ? "audio/webm"
          : "";
      const recorder = preferredMimeType
        ? new MediaRecorder(stream, { mimeType: preferredMimeType })
        : new MediaRecorder(stream);

      streamRef.current = stream;
      recorderRef.current = recorder;
      audioChunksRef.current = [];
      shouldSendRecordingRef.current = true;

      recorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      recorder.onerror = () => {
        setMicError("Jarvis microphone recording failed unexpectedly.");
        stopRecording();
      };

      recorder.onstop = () => {
        const mimeType = recorder.mimeType || preferredMimeType || "audio/webm";
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
        const shouldSendRecording = shouldSendRecordingRef.current;
        audioChunksRef.current = [];
        shouldSendRecordingRef.current = true;
        recorderRef.current = null;
        cleanupMediaStream();
        setIsRecording(false);

        if (shouldSendRecording && audioBlob.size > 0) {
          void sendRecordedAudio(audioBlob);
        }
      };

      recorder.start();
      setIsRecording(true);
    } catch (error: unknown) {
      cleanupMediaStream();
      recorderRef.current = null;
      setIsRecording(false);
      setMicError(
        error instanceof Error
          ? `Microphone access failed: ${error.message}`
          : "Microphone access failed."
      );
    }
  }

  function stopRecording(discardCurrent: boolean = false): void {
    const recorder = recorderRef.current;
    if (discardCurrent) {
      shouldSendRecordingRef.current = false;
    }

    if (!recorder) {
      cleanupMediaStream();
      setIsRecording(false);
      return;
    }

    if (recorder.state !== "inactive") {
      recorder.stop();
    } else {
      recorderRef.current = null;
      cleanupMediaStream();
      setIsRecording(false);
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    const trimmedValue = inputValue.trim();

    if (!trimmedValue) {
      return;
    }

    if (isRequestActive) {
      pushMessage({
        id: `busy-${Date.now()}`,
        role: "system",
        label: "System",
        content: "Jarvis is still handling the previous request. Please wait or press Interrupt."
      });
      return;
    }

    pushMessage({
      id: `user-${Date.now()}`,
      role: "user",
      label: "You",
      content: trimmedValue
    });

    const assistantMessageId = `assistant-http-${Date.now()}`;
    pushMessage({
      id: assistantMessageId,
      role: "assistant",
      label: "Jarvis",
      content: "",
      toolOutcome: undefined
    });

    clearAutoSpeakQueue();
    cleanupAudioPlayback();
    void streamChatOverHttp(trimmedValue, assistantMessageId);
    setInputValue("");
  }

  function interruptCurrentRequest(): void {
    if (!isRequestActive && !isRecording && socketStatus !== "connecting") {
      return;
    }

    stopRecording(true);
    chatAbortControllerRef.current?.abort();
    chatAbortControllerRef.current = null;
    activeHttpAssistantMessageIdRef.current = null;
    clearAutoSpeakQueue();
    cleanupAudioPlayback();
    activeStreamIdRef.current = null;
    activeStreamMessageIdRef.current = null;
    setPendingAssistantResponse(false);
    setIsTextStreaming(false);
    setAwaitingAssistantAudio(false);
    setTransportNotice("Interrupted. Reconnecting...");

    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }

    const socket = socketRef.current;
    if (!socket || socket.readyState === WebSocket.CLOSED) {
      reconnectTimerRef.current = window.setTimeout(() => {
        reconnectTimerRef.current = null;
        connect();
      }, 120);
    } else {
      manualDisconnectRef.current = true;
      socket.close(1000, "interrupt");
      socketRef.current = null;
      setSocketStatus("disconnected");
      reconnectTimerRef.current = window.setTimeout(() => {
        reconnectTimerRef.current = null;
        connect();
      }, 120);
    }

    pushMessage({
      id: `interrupt-${Date.now()}`,
      role: "system",
      label: "System",
      content: "Current request interrupted."
    });
  }

  const sendConfirmation = useCallback((approved: boolean) => {
    if (confirmCountdownRef.current !== null) {
      window.clearInterval(confirmCountdownRef.current);
      confirmCountdownRef.current = null;
    }
    setConfirmationRequest(null);
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    socket.send(
      JSON.stringify({
        type: approved ? "tool.confirm" : "tool.deny",
        payload: {},
        session_id: sessionId,
      } satisfies Envelope)
    );
  }, [sessionId]);

  return (
    <main className="minimal-root min-h-screen px-4 py-5 text-[#f5f5f5] sm:px-8">
      {currentMode === "performance" && pendingAssistantResponse && <div className="top-progress-bar" />}

      {/* ── Risk Gate Confirmation Modal ───────────────────────────────── */}
      <AnimatePresence>
        {confirmationRequest && (
          <motion.div
            key="confirm-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
            role="dialog"
            aria-modal="true"
            aria-labelledby="confirm-title"
          >
            <motion.div
              key="confirm-panel"
              initial={{ opacity: 0, scale: 0.93, y: 16 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.93, y: 16 }}
              transition={{ duration: 0.22, ease: "easeOut" }}
              className="mx-4 w-full max-w-md rounded-2xl border border-[#2a2a2a] bg-[#111] shadow-2xl"
            >
              {/* Header */}
              <div
                className={`flex items-center gap-3 rounded-t-2xl px-5 py-4 ${
                  confirmationRequest.risk_level === "high"
                    ? "border-b border-red-900/40 bg-red-950/30"
                    : "border-b border-amber-900/30 bg-amber-950/20"
                }`}
              >
                <span
                  className={`flex h-7 w-7 items-center justify-center rounded-full text-[14px] ${
                    confirmationRequest.risk_level === "high"
                      ? "bg-red-900/60 text-red-300"
                      : "bg-amber-900/50 text-amber-300"
                  }`}
                  aria-hidden="true"
                >
                  {confirmationRequest.risk_level === "high" ? "⚠" : "!"}
                </span>
                <div className="flex-1">
                  <p
                    id="confirm-title"
                    className="text-[13px] font-semibold uppercase tracking-widest text-[#d4d4d4]"
                  >
                    {confirmationRequest.risk_level === "high" ? "High-Risk Action" : "Confirm Action"}
                  </p>
                  <p className="mt-0.5 font-mono text-[11px] text-[#888]">
                    {confirmationRequest.tool_name}
                  </p>
                </div>
                <span
                  className={`rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-widest ${
                    confirmationRequest.risk_level === "high"
                      ? "bg-red-900/50 text-red-400"
                      : "bg-amber-900/40 text-amber-400"
                  }`}
                >
                  {confirmationRequest.risk_level}
                </span>
              </div>

              {/* Body */}
              <div className="px-5 py-4">
                <p className="text-[15px] font-medium text-[#e8e8e8]">
                  {confirmationRequest.human_label}
                </p>
                <p className="mt-2 text-[13px] leading-6 text-[#999]">
                  {confirmationRequest.message}
                </p>
              </div>

              {/* Footer */}
              <div className="flex items-center justify-between gap-3 rounded-b-2xl border-t border-[#222] px-5 py-3">
                <p className="font-mono text-[11px] text-[#666]">
                  Auto-cancels in {confirmCountdown}s
                </p>
                <div className="flex gap-3">
                  <button
                    id="confirm-deny-btn"
                    type="button"
                    onClick={() => sendConfirmation(false)}
                    className="rounded-lg border border-[#333] bg-[#1a1a1a] px-4 py-2 text-[13px] text-[#b0b0b0] transition hover:border-[#555] hover:text-white"
                  >
                    Cancel
                  </button>
                  <button
                    id="confirm-approve-btn"
                    type="button"
                    onClick={() => sendConfirmation(true)}
                    className={`rounded-lg px-4 py-2 text-[13px] font-semibold text-white transition ${
                      confirmationRequest.risk_level === "high"
                        ? "bg-red-700 hover:bg-red-600"
                        : "bg-amber-700 hover:bg-amber-600"
                    }`}
                  >
                    Confirm
                  </button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
      <motion.section
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, ease: "easeOut" }}
        className="mx-auto flex min-h-[calc(100vh-2.5rem)] w-full max-w-5xl flex-col rounded-2xl border border-[#222] bg-[#0f0f0f]"
      >
        <header className="flex items-center justify-between border-b border-[#222] px-4 py-3 sm:px-5">
          <div className="flex min-h-[20px] flex-col items-start gap-1">
            <div className="flex items-center gap-2">
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  socketStatus === "connected"
                    ? "bg-[#8ea7c4]"
                    : socketStatus === "connecting"
                      ? "bg-[#85725c]"
                      : "bg-[#565656]"
                }`}
              />
              {showModelContext || isSwitchingMode ? (
                <p className="text-[12px] text-[#a7a7a7]">
                  Model: <span className="font-mono text-[11px] text-[#d2d2d2]">{activeModel || "updating"}</span>
                </p>
              ) : null}
              {!showModelContext && !isSwitchingMode && transportNotice ? (
                <p className="text-[12px] text-[#8e8e8e]">{transportNotice}</p>
              ) : null}
            </div>
            <p className="text-[11px] text-[#7f7f7f]">
              GPU: {Math.round(telemetry.gpuLoad)}% | RAM: {telemetry.ramAvailableGb.toFixed(2)} GB | Disk I/O: {telemetry.diskIoMBps.toFixed(2)} MB/s
              {telemetry.telemetrySource !== "measured" ? ` (${telemetry.telemetrySource})` : ""}
            </p>
            {!capabilities.numBatchSupported || capabilities.hostOptimizerStatus !== "applied" ? (
              <p className="text-[10px] text-[#6f6f6f]">
                {capabilities.numBatchSupported ? "" : capabilities.numBatchWarning}
                {capabilities.hostOptimizerStatus !== "applied"
                  ? `${capabilities.numBatchSupported ? "" : " | "}Host optimizer: ${capabilities.hostOptimizerStatus}`
                  : ""}
              </p>
            ) : null}
          </div>
          <PowerController
            currentMode={currentMode}
            isSwitchingMode={isSwitchingMode}
            turboModeActive={telemetry.gpuLoad >= TURBO_MODE_THRESHOLD}
            onSwitch={(mode) => {
              void handleSwitchMode(mode);
            }}
          />
        </header>

        <section className="flex flex-1 flex-col">
          <div ref={messagesViewportRef} className="flex-1 overflow-y-auto px-5 py-7 sm:px-8">
            <div className="mx-auto max-w-3xl space-y-7">
              {messages.map((message) => (
                <motion.article
                  key={message.id}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="relative space-y-1"
                >
                  <div className="flex items-center gap-2">
                    <p className="text-[11px] uppercase tracking-[0.18em] text-[#7d7d7d]">{message.label}</p>
                    {message.role === "assistant" && message.toolOutcome ? (
                      <span
                        className={`rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.14em] ${
                          message.toolOutcome.verified === true
                            ? "bg-[#1f3a2a] text-[#a9f1c3]"
                            : message.toolOutcome.verified === false
                              ? "bg-[#3f2323] text-[#f6b3b3]"
                              : "bg-[#2f2f2f] text-[#cfcfcf]"
                        }`}
                        title={
                          message.toolOutcome.evidence
                            ? `${message.toolOutcome.tool_name}: ${message.toolOutcome.evidence}`
                            : `${message.toolOutcome.tool_name}: ${message.toolOutcome.status}`
                        }
                      >
                        {message.toolOutcome.verified === true
                          ? "verified"
                          : message.toolOutcome.verified === false
                            ? "unverified"
                            : "reported"}
                      </span>
                    ) : null}
                  </div>
                  <p
                    className={`whitespace-pre-wrap text-[15px] leading-7 ${
                      message.role === "user"
                        ? "text-white"
                        : message.role === "assistant"
                          ? "text-[#e0e0e0]"
                          : "font-mono text-[12px] leading-6 text-[#a5a5a5]"
                    }`}
                  >
                    {message.content}
                  </p>
                  {message.role === "assistant" && message.content.trim().length > 0 ? (
                    <button
                      type="button"
                      onClick={() => {
                        void handleManualListen(message.id, message.content);
                      }}
                      className={`absolute bottom-0 right-0 inline-flex items-center justify-center p-1 text-[#b8c2cf] transition hover:opacity-90 ${
                        manualListenMessageId === message.id && isPlayingAudio
                          ? "opacity-95 listening-pulse"
                          : "opacity-30"
                      }`}
                      aria-label="Listen to this reply"
                      title="Listen"
                    >
                      <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor" aria-hidden="true">
                        <path d="M14 3.23a1 1 0 0 1 1.52.85v15.84a1 1 0 0 1-1.7.72l-4.58-4.58H5a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1h4.24l4.58-4.58A1 1 0 0 1 14 3.23Zm4.28 2.28a1 1 0 0 1 1.41 0 9 9 0 0 1 0 12.73 1 1 0 1 1-1.41-1.41 7 7 0 0 0 0-9.91 1 1 0 0 1 0-1.41Zm-2.12 2.12a1 1 0 0 1 1.41 0 6 6 0 0 1 0 8.49 1 1 0 1 1-1.41-1.41 4 4 0 0 0 0-5.66 1 1 0 0 1 0-1.42Z" />
                      </svg>
                    </button>
                  ) : null}
                </motion.article>
              ))}
              {(pendingAssistantResponse || isTextStreaming || awaitingAssistantAudio) && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.32, ease: "easeOut" }}
                  className="thinking-shell"
                >
                  <div className="thinking-wave" aria-hidden="true">
                    <span className="thinking-dot" />
                    <span className="thinking-dot" />
                    <span className="thinking-dot" />
                  </div>
                  <p className="thinking-label">
                    {pendingAssistantResponse
                      ? "Jarvis is thinking..."
                      : isTextStreaming
                        ? "Jarvis is composing..."
                        : "Jarvis is preparing voice..."}
                  </p>
                </motion.div>
              )}
            </div>
          </div>

          {(statusLoadError || prewarmWarning || audioUiError || sttUiError || micError) && (
            <div className="border-t border-[#222] px-5 py-2 text-[12px] text-[#9f9f9f] sm:px-8">
              {statusLoadError || prewarmWarning || audioUiError || sttUiError || micError}
            </div>
          )}

          <div className="border-t border-[#222] px-5 py-3 sm:px-8">
            <form onSubmit={handleSubmit} className="flex items-center gap-3 border-b border-[#2a2a2a] pb-2">
              <input
                value={inputValue}
                onChange={(event) => setInputValue(event.target.value)}
                placeholder="Type a command..."
                autoComplete="off"
                className="h-10 flex-1 bg-transparent text-[15px] text-white outline-none placeholder:text-[#5e5e5e]"
              />
              <button
                type="button"
                onClick={() => {
                  const next = !voiceAutoSpeak;
                  setVoiceAutoSpeak(next);
                  persistVoiceAutoSpeak(next);
                }}
                className="text-[12px] text-[#9ea9b7] transition hover:text-[#d8e2f0]"
              >
                {voiceAutoSpeak ? "VOICE: ON" : "VOICE: OFF"}
              </button>
              <button
                type="submit"
                disabled={inputValue.trim().length === 0 || isRequestActive}
                className="text-[13px] text-[#d8d8d8] transition hover:text-white disabled:opacity-35"
              >
                Send
              </button>
              <button
                type="button"
                onClick={interruptCurrentRequest}
                disabled={!isRequestActive && !isRecording && socketStatus !== "connecting"}
                className="text-[13px] text-[#dca7a7] transition hover:text-[#ffd0d0] disabled:opacity-35"
              >
                Interrupt
              </button>
            </form>

            <div className="mt-2">
              <button
                type="button"
                onClick={() => setShowVoiceControls((current) => !current)}
                className="text-[12px] text-[#8f8f8f] transition hover:text-[#c6c6c6]"
              >
                {showVoiceControls ? "Hide voice controls" : "Voice controls"}
              </button>
            </div>

            {isRequestActive && (
              <div className="mt-2 flex items-center gap-2 text-[12px] text-[#8e8e8e]">
                <span className="h-1.5 w-1.5 rounded-full bg-[#8e8e8e] breathing-soft" />
                {pendingAssistantResponse
                  ? "Jarvis is generating a response..."
                  : isTextStreaming
                    ? "Jarvis is streaming a response..."
                  : awaitingAssistantAudio
                    ? "Jarvis is preparing voice response..."
                    : "Jarvis is streaming voice response..."}
              </div>
            )}

            {showVoiceControls && (
              <div className="mt-2 border-t border-[#222] pt-2">
                <div className="flex items-center justify-between gap-3">
                  <div className="inline-flex items-center gap-2 rounded-md border border-[#2a2a2a] px-3 py-1.5">
                    <span className={`h-2 w-2 rounded-full ${isRecording ? "listening-pulse bg-[#7f9fbf]" : "bg-[#4a4a4a]"}`} />
                    <span className="text-[12px] text-[#c9c9c9]">
                      {isRequestActive
                        ? "Busy... press Interrupt to force stop"
                        : isRecording
                          ? `Listening... release ${PTT_HOTKEY}`
                          : `Hold ${PTT_HOTKEY} to talk`}
                    </span>
                  </div>
                  <p className="text-[11px] font-mono text-[#7f7f7f]">
                    {micError ||
                      (mediaRecorderSupported
                        ? isRecording
                          ? "mic listening"
                          : "mic ready"
                        : "media recorder unavailable")}
                  </p>
                </div>
              </div>
            )}
          </div>
        </section>
      </motion.section>
    </main>
  );
}

export default App;
