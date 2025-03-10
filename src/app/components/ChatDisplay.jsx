import React from "react";

const ChatDisplay = ({ chatHistory }) => {
  return (
    <div className="max-w-2xl mx-auto p-4 bg-gray-100 rounded-lg shadow-md">
      <h2 className="text-lg font-bold mb-4">Chat History</h2>
      <div className="space-y-2">
        {Object.entries(chatHistory).map(([timestamp, message]) => (
          <div key={timestamp} className="p-2 bg-white rounded shadow-sm">
            <div className="text-xs text-gray-500">{timestamp}</div>
            <div>{message}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ChatDisplay;
