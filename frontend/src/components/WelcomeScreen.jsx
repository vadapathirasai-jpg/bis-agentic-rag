import React from 'react';
import { Smartphone, FileCheck2, HelpCircle, ShieldAlert, Sparkles, CheckCircle2 } from 'lucide-react';

const SUGGESTIONS = [
  {
    icon: Smartphone,
    title: "What is BIS CARE?",
    description: "Learn about the official mobile app, verifying licenses, and checking hallmarked gold.",
    query: "What is BIS CARE?",
  },
  {
    icon: FileCheck2,
    title: "How to file a complaint?",
    description: "Procedures for lodging complaints with CMED online, via email, or in person.",
    query: "How can a consumer file a complaint with BIS?",
  },
  {
    icon: HelpCircle,
    title: "What does the ISI mark mean?",
    description: "Understand the Scheme-I quality mark, manufacturer compliance, and consumer safety.",
    query: "What does the ISI mark mean?",
  },
  {
    icon: ShieldAlert,
    title: "How to verify HUID?",
    description: "Step-by-step guidance on verifying the 6-digit Hallmark Unique Identification code.",
    query: "How can HUID be verified by consumers?",
  },
];

export default function WelcomeScreen({ onSelectQuery }) {
  return (
    <div className="welcome-screen">
      <div className="welcome-hero">
        <div className="hero-pill">
          <Sparkles size={14} className="hero-pill-icon" />
          <span>Official BIS Citizen Knowledge Assistant</span>
        </div>
        <h2 className="welcome-title">
          Your intelligent guide to BIS standards and consumer services.
        </h2>
        <p className="welcome-desc">
          Ask questions in natural language and get factual answers grounded strictly in
          official Bureau of Indian Standards documentation, regulations, and consumer advisories.
        </p>

        <div className="hero-features">
          <div className="feature-chip">
            <CheckCircle2 size={15} className="chip-icon" />
            <span>Authoritative BIS Evidence</span>
          </div>
          <div className="feature-chip">
            <CheckCircle2 size={15} className="chip-icon" />
            <span>Direct Source Citations</span>
          </div>
          <div className="feature-chip">
            <CheckCircle2 size={15} className="chip-icon" />
            <span>Zero Hallucination Guarantee</span>
          </div>
        </div>
      </div>

      <div className="suggestions-container">
        <h3 className="suggestions-heading">Suggested Inquiries for Citizens</h3>
        <div className="suggestions-grid">
          {SUGGESTIONS.map((item, index) => {
            const Icon = item.icon;
            return (
              <button
                key={index}
                type="button"
                className="suggestion-card"
                onClick={() => onSelectQuery(item.query)}
              >
                <div className="card-header">
                  <div className="card-icon-box">
                    <Icon size={18} />
                  </div>
                  <span className="card-arrow">↗</span>
                </div>
                <h4 className="card-title">{item.title}</h4>
                <p className="card-desc">{item.description}</p>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
