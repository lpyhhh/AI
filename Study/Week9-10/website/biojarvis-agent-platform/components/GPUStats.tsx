
import React from 'react';

interface GPUStatsProps {
  used: number;
  total: number;
  model: string;
}

export const GPUStats: React.FC<GPUStatsProps> = ({ used, total, model }) => {
  const percentage = (used / total) * 100;
  
  return (
    <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700 backdrop-blur-sm">
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-sm font-semibold text-slate-300">NVIDIA L4 Status</h3>
        <span className="text-[10px] uppercase tracking-wider text-emerald-400 font-bold bg-emerald-400/10 px-2 py-0.5 rounded">Online</span>
      </div>
      
      <div className="space-y-4">
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-slate-400 text-[11px]">VRAM Usage</span>
            <span className="text-slate-200 font-mono">{used.toFixed(1)}GB / {total}GB</span>
          </div>
          <div className="h-2 w-full bg-slate-900 rounded-full overflow-hidden">
            <div 
              className="h-full bg-emerald-500 rounded-full transition-all duration-1000 ease-out"
              style={{ width: `${percentage}%` }}
            />
          </div>
        </div>
        
        <div className="pt-2 border-t border-slate-700/50">
          <div className="flex justify-between text-[11px] mb-1">
            <span className="text-slate-400">Active Architecture</span>
            <span className="text-slate-200">{model}</span>
          </div>
          <div className="flex justify-between text-[11px]">
            <span className="text-slate-400">Quantization</span>
            <span className="text-slate-200">4-bit (bitsandbytes)</span>
          </div>
        </div>
      </div>
    </div>
  );
};
