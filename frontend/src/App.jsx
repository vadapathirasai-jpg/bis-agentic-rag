import React, { useState, useEffect, useRef } from 'react';
import Header from './components/Header';
import WelcomeScreen from './components/WelcomeScreen';
import ChatMessage from './components/ChatMessage';
import LoadingIndicator from './components/LoadingIndicator';
import ChatInput from './components/ChatInput';
import { checkStatus, sendChatMessage } from './services/api';
import { AlertCircle } from 'lucide-react';

export default function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState(null);
  const [generalError, setGeneralError] = useState(null);
  const messagesEndRef = useRef(null);

  // Poll status on mount
  useEffect(() => {
    const fetchStatus = async () => {
      const status = await checkStatus();
      setBackendStatus(status);
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  // Auto-scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleNewChat = () => {
    setMessages([]);
    setGeneralError(null);
  };

  const handleSendMessage = async (queryText) => {
    if (!queryText.trim() || isLoading) return;

    setGeneralError(null);
    const userMsg = { role: 'user', content: queryText };
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const data = await sendChatMessage(queryText);
      const assistantMsg = { role: 'assistant', data };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      console.error('Chat request error:', err);
      const errorMsg = {
        role: 'assistant',
        data: {
          answer: null,
          status_message: "Sorry, I couldn't process that request right now. Please verify that the backend server is running and try again.",
          evidence: [],
        },
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <Header onNewChat={handleNewChat} backendStatus={backendStatus} />

      <main className="main-content">
        <div className="chat-viewport">
          {messages.length === 0 ? (
            <WelcomeScreen onSelectQuery={handleSendMessage} />
          ) : (
            <div className="messages-stream">
              {messages.map((msg, index) => (
                <ChatMessage key={index} message={msg} />
              ))}
              {isLoading && <LoadingIndicator />}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </main>

      <footer className="site-footer">
        <div className="footer-container">
          {generalError && (
            <div className="global-error-banner">
              <AlertCircle size={16} />
              <span>{generalError}</span>
            </div>
          )}
          <ChatInput onSendMessage={handleSendMessage} disabled={isLoading} />
        </div>
      </footer>
    </div>
  );
}
