
import React from 'react';
import { ProteinAnalysisResult } from '../types';

interface ProteinViewProps {
  data: ProteinAnalysisResult;
}

export const ProteinView: React.FC<ProteinViewProps> = ({ data }) => {
  return (
    <div className="my-4 bg-slate-900/80 border border-emerald-500/30 rounded-xl overflow-hidden shadow-lg shadow-emerald-500/10">
      <div className="bg-emerald-500/10 px-4 py-2 border-b border-emerald-500/30 flex justify-between items-center">
        <span className="text-emerald-400 text-xs font-bold uppercase tracking-widest">ESM-2 LoRA Inference Result</span>
        <span className="text-emerald-400 font-mono text-xs">{(data.confidence * 100).toFixed(1)}% Confidence</span>
      </div>
      
      <div className="p-4 space-y-4">
        <div className="bg-slate-950 p-3 rounded border border-slate-800 font-mono text-xs break-all leading-relaxed text-slate-300">
          <span className="text-emerald-500 mr-2">SEQ:</span>
          {data.sequence}
        </div>
        
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1">
            <span className="text-slate-500 text-[10px] uppercase font-bold">Toxicity Potential</span>
            <div className="flex items-center gap-2">
              <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div className="h-full bg-red-500" style={{ width: `${data.toxicityScore * 100}%` }} />
              </div>
              <span className="text-xs text-red-400 font-bold">{(data.toxicityScore * 100).toFixed(0)}%</span>
            </div>
          </div>
          
          <div className="space-y-1">
            <span className="text-slate-500 text-[10px] uppercase font-bold">Folding Stability</span>
            <div className="flex items-center gap-2">
              <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div className="h-full bg-blue-500" style={{ width: `${data.stability * 100}%` }} />
              </div>
              <span className="text-xs text-blue-400 font-bold">{(data.stability * 100).toFixed(0)}%</span>
            </div>
          </div>
        </div>

        <div className="space-y-2">
          <span className="text-slate-500 text-[10px] uppercase font-bold">Structural Annotations</span>
          <ul className="text-xs space-y-1">
            {data.structuralInsights.map((insight, i) => (
              <li key={i} className="flex items-start gap-2 text-slate-300">
                <span className="text-emerald-500 mt-1">•</span>
                {insight}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};
