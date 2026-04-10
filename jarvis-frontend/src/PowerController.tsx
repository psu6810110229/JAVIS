type PowerMode = "eco" | "performance";

type PowerControllerProps = {
  currentMode: PowerMode;
  isSwitchingMode: boolean;
  turboModeActive?: boolean;
  onSwitch: (mode: PowerMode) => void;
};

function PowerController({
  currentMode,
  isSwitchingMode,
  turboModeActive = false,
  onSwitch
}: PowerControllerProps) {
  const isEco = currentMode === "eco";

  return (
    <section className="flex items-center gap-2 rounded-full border border-[#222] bg-[#121212] p-1">
      <div className="px-2 text-[11px] text-[#a0a0a0]">Mode</div>
      <div className="flex items-center gap-1">
        <button
          type="button"
          onClick={() => onSwitch("eco")}
          disabled={isSwitchingMode || isEco}
          className={`rounded-full px-3 py-1.5 text-[12px] transition duration-[400ms] ease-in-out disabled:cursor-not-allowed ${
            isEco
              ? "bg-[#1d2738] text-[#d8e2f0] breathing-soft"
              : "text-[#8a8a8a] hover:text-[#d8e2f0]"
          }`}
        >
          <span className="inline-flex items-center gap-2">
            <span className="h-1.5 w-1.5 rounded-full bg-[#5e6d8d]" />
            Eco
          </span>
        </button>

        <button
          type="button"
          onClick={() => onSwitch("performance")}
          disabled={isSwitchingMode || !isEco}
          className={`rounded-full px-3 py-1.5 text-[12px] transition duration-[400ms] ease-in-out disabled:cursor-not-allowed ${
            !isEco
              ? "bg-[#2a2118] text-[#f0e2cd] breathing-soft"
              : "text-[#8a8a8a] hover:text-[#f0e2cd]"
          }`}
        >
          <span className="inline-flex items-center gap-2">
            <span className="h-1.5 w-1.5 rounded-full bg-[#b7874e]" />
            Performance
          </span>
        </button>
      </div>

      {isSwitchingMode && (
        <div className="px-2 text-[11px] text-[#8f8f8f]">Reconfiguring...</div>
      )}

      {!isSwitchingMode && (
        <div className={`px-2 text-[11px] ${turboModeActive ? "text-[#e9d8b2]" : "text-[#6f6f6f]"}`}>
          {turboModeActive ? "Turbo Mode: ACTIVE" : "Turbo Mode: standby"}
        </div>
      )}
    </section>
  );
}

export default PowerController;
