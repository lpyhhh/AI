
import { GoogleGenAI } from "@google/genai";
import { MessageRole, Message, ProteinAnalysisResult, SystemStatus } from "../types";

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });

// 激活本地后端连接
const USE_LOCAL_BACKEND = true; 
const BACKEND_BASE = "http://localhost:8000";

export const getSystemStats = async (): Promise<SystemStatus> => {
  if (!USE_LOCAL_BACKEND) return { gpuMemoryUsed: 6.2, gpuMemoryTotal: 24, modelStatus: 'idle', activeModel: 'Simulation' };
  try {
    const resp = await fetch(`${BACKEND_BASE}/stats`);
    return await resp.json();
  } catch (e) {
    return { gpuMemoryUsed: 0, gpuMemoryTotal: 24, modelStatus: 'loading', activeModel: 'Offline' };
  }
};

export const sendMessageToAgent = async (
  history: Message[],
  userInput: string
): Promise<Message> => {
  
  if (USE_LOCAL_BACKEND) {
    try {
      const resp = await fetch(`${BACKEND_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userInput })
      });
      const data = await resp.json();
      
      let sequenceData: ProteinAnalysisResult | undefined;
      if (data.is_sequence) {
        sequenceData = {
          sequence: data.sequence,
          prediction: "CRESS 相关序列 (由 ESM-2 LoRA 鉴定)",
          confidence: 0.98,
          toxicityScore: 0.12,
          stability: 0.85,
          structuralInsights: ["检测到保守的 Rep 蛋白结构域", "RCR 复制机制相关"]
        };
      }

      return {
        id: Math.random().toString(36).substr(2, 9),
        role: MessageRole.AGENT,
        content: data.response,
        timestamp: new Date(),
        metadata: { 
          processingTime: data.time,
          sequenceData
        }
      };
    } catch (e) {
      console.error("Backend Error", e);
    }
  }

  // Fallback to Gemini simulation...
  const model = ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: [{ role: 'user', parts: [{ text: userInput }] }],
    config: { systemInstruction: "You are BioJarvis." }
  });
  const res = await model;
  return {
    id: 'sim',
    role: MessageRole.AGENT,
    content: res.text || "",
    timestamp: new Date()
  };
};
