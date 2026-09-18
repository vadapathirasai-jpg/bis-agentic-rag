import React, { useEffect, useState } from 'react';
import { Shield, Loader2 } from 'lucide-react';

const STAGES = [
  "Searching official BIS sources...",
  "Analyzing retrieved evidence...",
  "Preparing your answer...",
];

export default function LoadingIndicator() {
  const [stageIndex, setStageIndex] = useState(0);

  useEffect(() => {
    const timer1 = setTimeout(() => setStageIndex(1), 1200);
    const timer2 = setTimeout(() => setStageIndex(2), 2600);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
    };
  }, []);

  return (
    <div className="chat-row assistant-row loading-row">
      <div className="avatar assistant-avatar" title="BIS Sahayak">
        <Shield size={18} />
      </div>

      <div className="loading-bubble">
        <div className="loading-content">
          <Loader2 className="loading-spinner" size={18} />
          <span className="loading-text">{STAGES[stageIndex]}</span>
        </div>
      </div>
    </div>
  );
}
