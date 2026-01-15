
export enum MessageRole {
  USER = 'user',
  AGENT = 'agent',
  SYSTEM = 'system'
}

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: Date;
  metadata?: {
    toolCall?: string;
    sequenceData?: ProteinAnalysisResult;
    processingTime?: number;
  };
}

export interface ProteinAnalysisResult {
  sequence: string;
  prediction: string;
  confidence: number;
  toxicityScore: number;
  stability: number;
  structuralInsights: string[];
}

export interface SystemStatus {
  gpuMemoryUsed: number;
  gpuMemoryTotal: number;
  modelStatus: 'idle' | 'processing' | 'loading';
  activeModel: string;
}
