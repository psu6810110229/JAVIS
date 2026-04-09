import { FormEvent, useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";

type SocketStatus = "disconnected" | "connecting" | "connected";

type Envelope = {
  type: string;
  payload: Record<string, unknown>;
  timestamp?: string;
  session_id?: string;
};

type ChatMessage = {
  id: string;
  role: "system" | "user" | "assistant";
  label: string;
  content: string;
};

type AssistantAudioPayload = {
  audio_base64: string;
  mime_type: string;
  voice: string;
};

const DEFAULT_WS_URL = "ws://127.0.0.1:8000/ws";

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
  const socketRef = useRef<WebSocket | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<BlobPart[]>([]);
  const shouldSendRecordingRef = useRef<boolean>(true);
  const activeAudioRef = useRef<HTMLAudioElement | null>(null);
  const activeAudioUrlRef = useRef<string>("");
  const wsUrl = import.meta.env.VITE_JARVIS_WS_URL ?? DEFAULT_WS_URL;
  const mediaRecorderSupported =
    typeof window !== "undefined" &&
    typeof navigator !== "undefined" &&
    typeof navigator.mediaDevices !== "undefined" &&
    typeof navigator.mediaDevices.getUserMedia === "function" &&
    typeof MediaRecorder !== "undefined";

  useEffect(() => {
    return () => {
      stopRecording(true);
      cleanupAudioPlayback();
      socketRef.current?.close();
    };
  }, []);

  function pushMessage(message: ChatMessage): void {
    setMessages((current) => [...current, message]);
  }

  function cleanupMediaStream(): void {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }

  function cleanupAudioPlayback(): void {
    if (activeAudioRef.current) {
      activeAudioRef.current.pause();
      activeAudioRef.current.src = "";
      activeAudioRef.current = null;
    }

    if (activeAudioUrlRef.current) {
      URL.revokeObjectURL(activeAudioUrlRef.current);
      activeAudioUrlRef.current = "";
    }

    setIsPlayingAudio(false);
  }

  async function playAssistantAudio(payload: AssistantAudioPayload): Promise<void> {
    cleanupAudioPlayback();

    try {
      const audioBlob = base64ToBlob(payload.audio_base64, payload.mime_type);
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);

      activeAudioRef.current = audio;
      activeAudioUrlRef.current = audioUrl;
      setIsPlayingAudio(true);

      audio.onended = () => {
        cleanupAudioPlayback();
      };
      audio.onerror = () => {
        cleanupAudioPlayback();
        pushMessage({
          id: `audio-playback-error-${Date.now()}`,
          role: "system",
          label: "Audio",
          content: "Jarvis could not play the assistant audio reply."
        });
      };

      await audio.play();
    } catch (error: unknown) {
      cleanupAudioPlayback();
      pushMessage({
        id: `audio-playback-exception-${Date.now()}`,
        role: "system",
        label: "Audio",
        content:
          error instanceof Error
            ? error.message
            : "Jarvis encountered an unexpected audio playback error."
      });
    }
  }

  function connect(): void {
    if (socketRef.current?.readyState === WebSocket.OPEN || socketStatus === "connecting") {
      return;
    }

    setSocketStatus("connecting");
    const socket = new WebSocket(wsUrl);
    socketRef.current = socket;

    socket.onopen = () => {
      setSocketStatus("connected");
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

      if (data.type === "speech.transcript") {
        const transcriptText = typeof data.payload.text === "string" ? data.payload.text : "";
        if (transcriptText) {
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
          void playAssistantAudio(payload as AssistantAudioPayload);
        } else {
          pushMessage({
            id: `assistant-audio-invalid-${Date.now()}`,
            role: "system",
            label: "Audio",
            content: "Jarvis returned an invalid assistant audio payload."
          });
        }
        return;
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
      pushMessage({
        id: `socket-error-${Date.now()}`,
        role: "system",
        label: "Error",
        content: "The Jarvis backend connection failed."
      });
    };

    socket.onclose = () => {
      stopRecording(true);
      cleanupAudioPlayback();
      setSocketStatus("disconnected");
      socketRef.current = null;
    };
  }

  function disconnect(): void {
    stopRecording(true);
    cleanupAudioPlayback();

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
        content: "Connect to Jarvis before sending microphone audio."
      });
      return;
    }

    try {
      const audioBase64 = await blobToBase64(audioBlob);
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
    const socket = socketRef.current;
    const trimmedValue = inputValue.trim();

    if (!socket || socket.readyState !== WebSocket.OPEN || !trimmedValue) {
      return;
    }

    pushMessage({
      id: `user-${Date.now()}`,
      role: "user",
      label: "You",
      content: trimmedValue
    });

    socket.send(
      JSON.stringify({
        type: "text.input",
        payload: {
          text: trimmedValue
        },
        session_id: sessionId
      } satisfies Envelope)
    );

    setInputValue("");
  }

  return (
    <main className="min-h-screen px-4 py-6 text-ink sm:px-8">
      <motion.section
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45, ease: "easeOut" }}
        className="mx-auto flex min-h-[calc(100vh-3rem)] w-full max-w-6xl flex-col gap-6 rounded-[32px] border border-white/10 bg-white/5 p-6 shadow-glow backdrop-blur-xl"
      >
        <header className="flex flex-col gap-4 border-b border-white/10 pb-6 md:flex-row md:items-end md:justify-between">
          <div className="space-y-3">
            <p className="text-sm uppercase tracking-[0.35em] text-accent/80">Voice AI Assistant</p>
            <div>
              <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">Jarvis</h1>
              <p className="mt-2 max-w-2xl text-sm text-slate-300 sm:text-base">
                Phase 2 adds push-to-talk voice input, Thai speech synthesis, and native audio playback
                over the existing FastAPI WebSocket session.
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <span
              className={`rounded-full px-4 py-2 text-xs font-semibold uppercase tracking-[0.24em] ${
                socketStatus === "connected"
                  ? "bg-emerald-500/15 text-emerald-200"
                  : socketStatus === "connecting"
                    ? "bg-amber-500/15 text-amber-200"
                    : "bg-slate-500/15 text-slate-200"
              }`}
            >
              {socketStatus}
            </span>
            <span
              className={`rounded-full px-4 py-2 text-xs font-semibold uppercase tracking-[0.24em] ${
                isPlayingAudio ? "bg-cyan-500/15 text-cyan-200" : "bg-slate-500/15 text-slate-300"
              }`}
            >
              {isPlayingAudio ? "playing audio" : "audio idle"}
            </span>
            <button
              type="button"
              onClick={connect}
              className="rounded-full bg-accent px-5 py-3 text-sm font-semibold text-slate-950 transition hover:scale-[1.01]"
            >
              Connect
            </button>
            <button
              type="button"
              onClick={disconnect}
              className="rounded-full border border-white/15 px-5 py-3 text-sm font-semibold text-white transition hover:bg-white/10"
            >
              Disconnect
            </button>
          </div>
        </header>

        <section className="grid flex-1 gap-6 lg:grid-cols-[1.5fr_0.8fr]">
          <div className="flex min-h-[28rem] flex-col rounded-[28px] border border-white/10 bg-panel/80 p-4">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-semibold">Conversation</h2>
              <span className="text-xs uppercase tracking-[0.3em] text-slate-400">
                Session {sessionId || "pending"}
              </span>
            </div>

            <div className="flex-1 space-y-3 overflow-y-auto pr-2">
              {messages.map((message) => (
                <motion.article
                  key={message.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`rounded-3xl px-4 py-3 ${
                    message.role === "user"
                      ? "ml-auto max-w-[80%] bg-accent text-slate-950"
                      : message.role === "assistant"
                        ? "max-w-[85%] bg-white/8 text-white"
                        : "max-w-[85%] border border-white/10 bg-white/5 text-slate-200"
                  }`}
                >
                  <p className="mb-2 text-[11px] uppercase tracking-[0.28em] opacity-70">{message.label}</p>
                  <p className="whitespace-pre-wrap text-sm leading-6">{message.content}</p>
                </motion.article>
              ))}
            </div>

            <div className="mt-4 grid gap-3 md:grid-cols-[1fr_auto]">
              <button
                type="button"
                disabled={socketStatus !== "connected" || !mediaRecorderSupported}
                onPointerDown={() => {
                  void startRecording();
                }}
                onPointerUp={() => {
                  stopRecording();
                }}
                onPointerLeave={() => {
                  stopRecording();
                }}
                onPointerCancel={() => {
                  stopRecording();
                }}
                className={`touch-none rounded-3xl px-6 py-4 text-left text-sm font-semibold transition ${
                  isRecording
                    ? "bg-rose-500 text-white shadow-lg shadow-rose-500/30"
                    : "border border-white/10 bg-slate-950/60 text-white hover:bg-slate-900/80"
                } disabled:cursor-not-allowed disabled:opacity-40`}
              >
                <span className="block text-[11px] uppercase tracking-[0.3em] opacity-70">Push To Talk</span>
                <span className="mt-2 block text-base">
                  {isRecording ? "Recording... release to send" : "Hold to record and send one voice turn"}
                </span>
              </button>

              <div className="rounded-3xl border border-white/10 bg-slate-950/40 px-4 py-3 text-sm text-slate-300">
                <p className="text-[11px] uppercase tracking-[0.28em] text-accent/80">Mic Status</p>
                <p className="mt-2">
                  {micError ||
                    (mediaRecorderSupported
                      ? isRecording
                        ? "Microphone is capturing WebM audio."
                        : "Microphone ready."
                      : "MediaRecorder is unavailable in this runtime.")}
                </p>
              </div>
            </div>

            <form onSubmit={handleSubmit} className="mt-4 flex flex-col gap-3 sm:flex-row">
              <textarea
                value={inputValue}
                onChange={(event) => setInputValue(event.target.value)}
                placeholder="Send a message to Jarvis..."
                rows={3}
                className="min-h-[88px] flex-1 rounded-3xl border border-white/10 bg-slate-950/50 px-4 py-3 text-sm text-white outline-none ring-0 placeholder:text-slate-500"
              />
              <button
                type="submit"
                disabled={socketStatus !== "connected" || inputValue.trim().length === 0}
                className="rounded-3xl bg-white px-6 py-3 text-sm font-semibold text-slate-950 transition disabled:cursor-not-allowed disabled:opacity-40"
              >
                Send
              </button>
            </form>
          </div>

          <aside className="flex flex-col gap-4 rounded-[28px] border border-white/10 bg-white/6 p-5">
            <div className="rounded-3xl border border-white/10 bg-slate-950/40 p-4">
              <p className="text-xs uppercase tracking-[0.28em] text-accent/80">Backend</p>
              <p className="mt-3 text-sm text-slate-300">{wsUrl}</p>
            </div>
            <div className="rounded-3xl border border-white/10 bg-slate-950/40 p-4">
              <p className="text-xs uppercase tracking-[0.28em] text-accent/80">Supported Events</p>
              <ul className="mt-3 space-y-2 text-sm text-slate-200">
                <li><code>session.start</code> / <code>session.end</code></li>
                <li><code>text.input</code> / <code>text.output</code></li>
                <li><code>audio.chunk</code> / <code>audio.ack</code></li>
                <li><code>speech.transcript</code> / <code>assistant_audio</code></li>
                <li><code>error</code></li>
              </ul>
            </div>
            <div className="rounded-3xl border border-white/10 bg-slate-950/40 p-4">
              <p className="text-xs uppercase tracking-[0.28em] text-accent/80">Voice Loop</p>
              <p className="mt-3 text-sm leading-6 text-slate-300">
                The frontend records one WebM utterance at a time, the backend transcribes Thai speech,
                queries Gemini, and returns both assistant text and Thai TTS audio.
              </p>
            </div>
          </aside>
        </section>
      </motion.section>
    </main>
  );
}

export default App;
