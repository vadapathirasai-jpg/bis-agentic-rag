import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { User, Shield, ExternalLink, ChevronDown, ChevronUp, Layers, CheckCircle2, RefreshCw } from 'lucide-react';

export default function ChatMessage({ message }) {
  const isUser = message.role === 'user';
  const [showTransparency, setShowTransparency] = useState(false);

  if (isUser) {
    return (
      <div className="chat-row user-row">
        <div className="user-bubble">
          <p className="user-text">{message.content}</p>
        </div>
        <div className="avatar user-avatar" title="You">
          <User size={16} />
        </div>
      </div>
    );
  }

  const {
    answer,
    evidence = [],
    retrieval_attempts = 1,
    evidence_sufficient = true,
    evaluation,
    query_refinements = [],
    model_used,
    status_message,
  } = message.data || {};

  return (
    <div className="chat-row assistant-row">
      <div className="avatar assistant-avatar" title="BIS Sahayak">
        <Shield size={18} />
      </div>

      <div className="assistant-content">
        <div className="assistant-bubble">
          {answer ? (
            <div className="markdown-body">
              <ReactMarkdown>{answer}</ReactMarkdown>
            </div>
          ) : (
            <div className="notice-box">
              <p className="notice-text">
                {status_message && !status_message.includes('failed') && !status_message.includes('error') && !status_message.includes('Google GenAI')
                  ? status_message
                  : "Sorry, I couldn't process that request right now. Please try again."}
              </p>
            </div>
          )}

          {/* RAG Transparency Collapsible */}
          <div className="transparency-section">
            <button
              type="button"
              className="transparency-toggle"
              onClick={() => setShowTransparency(!showTransparency)}
              aria-expanded={showTransparency}
            >
              <div className="toggle-left">
                <Layers size={14} />
                <span>How this answer was found</span>
              </div>
              {showTransparency ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>

            {showTransparency && (
              <div className="transparency-details">
                <div className="metric-item">
                  <span className="metric-label">Retrieval Attempts:</span>
                  <span className="metric-value">{retrieval_attempts}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Evidence Chunks Analyzed:</span>
                  <span className="metric-value">{evidence.length}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Evidence Sufficient:</span>
                  <span className={`metric-value ${evidence_sufficient ? 'badge-success' : 'badge-warning'}`}>
                    {evidence_sufficient ? (
                      <>
                        <CheckCircle2 size={12} className="inline-icon" /> Yes
                      </>
                    ) : (
                      'No'
                    )}
                  </span>
                </div>
                {evaluation?.reason && (
                  <div className="metric-item">
                    <span className="metric-label">Evaluation:</span>
                    <span className="metric-value text-muted">{evaluation.reason}</span>
                  </div>
                )}
                {model_used && (
                  <div className="metric-item">
                    <span className="metric-label">Model:</span>
                    <span className="metric-value">{model_used}</span>
                  </div>
                )}
                {query_refinements && query_refinements.length > 0 && (
                  <div className="refinement-box">
                    <div className="refinement-header">
                      <RefreshCw size={12} className="inline-icon spin" />
                      <span>Query Refined for Precision:</span>
                    </div>
                    <ul className="refinement-list">
                      {query_refinements.map((ref, idx) => (
                        <li key={idx}>"{ref}"</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Official BIS Evidence Sources Section */}
        {evidence && evidence.length > 0 && (
          <div className="sources-container">
            <div className="sources-header">
              <span className="sources-title">Official BIS Sources ({evidence.length})</span>
              <span className="sources-tag">Verified by Qdrant</span>
            </div>

            <div className="sources-grid">
              {evidence.map((item, idx) => {
                const scorePercent = item.score ? Math.round(item.score * 100) : null;
                return (
                  <div key={item.chunk_id || idx} className="source-card">
                    <div className="source-meta">
                      <span className="source-domain">bis.gov.in</span>
                      {item.chunk_id && (
                        <span className="source-chunk-badge" title={`Chunk ID: ${item.chunk_id}`}>
                          #{item.chunk_id.slice(0, 8)}
                        </span>
                      )}
                      {scorePercent && (
                        <span className="source-score" title="Cosine similarity match score">
                          {scorePercent}% match
                        </span>
                      )}
                    </div>
                    <h5 className="source-doc-title">{item.title || "Official BIS Resource"}</h5>
                    {item.section && <p className="source-section">{item.section}</p>}
                    {item.text && (
                      <p className="source-text-snippet">
                        "{item.text.length > 140 ? item.text.slice(0, 140) + '...' : item.text}"
                      </p>
                    )}

                    {item.source_url && (
                      <a
                        href={item.source_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="source-link"
                      >
                        <span>View official BIS source</span>
                        <ExternalLink size={12} />
                      </a>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
