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

const DEFAULT_WS_URL = "ws://127.0.0.1:8000/ws";

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
  const socketRef = useRef<WebSocket | null>(null);
  const wsUrl = import.meta.env.VITE_JARVIS_WS_URL ?? DEFAULT_WS_URL;

  useEffect(() => {
    return () => {
      socketRef.current?.close();
    };
  }, []);

  function pushMessage(message: ChatMessage): void {
    setMessages((current) => [...current, message]);
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

      if (data.type === "text.received") {
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
      setSocketStatus("disconnected");
      socketRef.current = null;
    };
  }

  function disconnect(): void {
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
                Minimal Tauri shell for the FastAPI backend. Phase 1 validates transport, session flow,
                and text responses over WebSocket.
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
                <li><code>error</code></li>
              </ul>
            </div>
            <div className="rounded-3xl border border-white/10 bg-slate-950/40 p-4">
              <p className="text-xs uppercase tracking-[0.28em] text-accent/80">Phase 1</p>
              <p className="mt-3 text-sm leading-6 text-slate-300">
                This shell validates the session contract and text workflow. Audio transport is accepted as
                base64 chunks, but microphone capture and real-time speech processing remain out of scope here.
              </p>
            </div>
          </aside>
        </section>
      </motion.section>
    </main>
  );
}

export default App;
