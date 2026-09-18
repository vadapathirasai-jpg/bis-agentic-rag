import React, { useState, useRef, useEffect } from 'react';
import { Send } from 'lucide-react';

export default function ChatInput({ onSendMessage, disabled }) {
  const [text, setText] = useState('');
  const textareaRef = useRef(null);

  // Auto-resize textarea height as content grows
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 140)}px`;
    }
  }, [text]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!text.trim() || disabled) return;
    onSendMessage(text);
    setText('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="chat-input-wrapper">
      <form className="chat-input-form" onSubmit={handleSubmit}>
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about BIS standards, hallmarking, ISI mark, or complaints..."
          rows={1}
          disabled={disabled}
          className="chat-textarea"
          aria-label="Your inquiry for BIS Sahayak"
        />

        <button
          type="submit"
          disabled={disabled || !text.trim()}
          className="btn-send"
          title="Send inquiry (Enter)"
          aria-label="Send message"
        >
          <Send size={16} />
        </button>
      </form>
      <div className="input-footer-note">
        <span>Press <strong>Enter</strong> to send, <strong>Shift + Enter</strong> for new line. Grounded in official BIS regulations.</span>
      </div>
    </div>
  );
}
