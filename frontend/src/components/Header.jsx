import React from 'react';
import { Shield, RotateCcw, Activity } from 'lucide-react';

export default function Header({ onNewChat, backendStatus }) {
  const isOnline = backendStatus && backendStatus.api && backendStatus.qdrant;

  return (
    <header className="site-header">
      <div className="header-container">
        <div className="header-brand">
          <div className="emblem-badge" aria-label="Bureau of Indian Standards Emblem">
            <Shield className="emblem-icon" size={24} />
          </div>
          <div className="brand-titles">
            <div className="brand-row">
              <h1 className="brand-name">BIS SAHAYAK</h1>
              <span className="gov-badge">GOVT OF INDIA</span>
            </div>
            <p className="brand-subtitle">
              AI-Powered Intelligent Assistant for Indian Standards & BIS Services
            </p>
          </div>
        </div>

        <div className="header-actions">
          <div className="status-pill" title={isOnline ? "Backend and Qdrant connected" : "Connecting to backend..."}>
            <span className={`status-dot ${isOnline ? 'online' : 'offline'}`} />
            <span className="status-text">
              {isOnline ? 'BIS Sahayak AI • Online' : 'Connecting to Backend...'}
            </span>
          </div>

          <button
            type="button"
            className="btn-new-chat"
            onClick={onNewChat}
            title="Start a new conversation"
            aria-label="Start new chat"
          >
            <RotateCcw size={15} />
            <span>New Chat</span>
          </button>
        </div>
      </div>
    </header>
  );
}
