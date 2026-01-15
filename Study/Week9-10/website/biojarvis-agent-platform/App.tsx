
import React, { useState, useRef, useEffect } from 'react';
import { Message, MessageRole, SystemStatus } from './types';
import { GPUStats } from './components/GPUStats';
import { ProteinView } from './components/ProteinView';
import { sendMessageToAgent, getSystemStats } from './services/bioService';

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: MessageRole.AGENT,
      content: "系统就绪。我已经将 Llama-3-8B 和 ESM-2 适配器挂载到 L4 显卡。请问需要分析什么序列？",
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    gpuMemoryUsed: 0,
    gpuMemoryTotal: 24.0,
    modelStatus: 'loading',
    activeModel: 'Initializing...'
  });

  const scrollRef = useRef<HTMLDivElement>(null);

  // 定时获取 GPU 状态
  useEffect(() => {
    const timer = setInterval(async () => {
      const stats = await getSystemStats();
      setSystemStatus(stats);
    }, 3000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: MessageRole.USER,
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await sendMessageToAgent(messages, input);
      setMessages(prev => [...prev, response]);
    } catch (error) {
      setMessages(prev => [...prev, {
        id: 'err',
        role: MessageRole.AGENT,
        content: "API 连接失败，请确保 AWS 8000 端口已转发且 fastapi_server.py 正在运行。",
        timestamp: new Date()
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-slate-950 overflow-hidden font-sans text-slate-200">
      <aside className="w-80 bg-slate-900 border-r border-slate-800 flex flex-col p-6">
        <div className="flex items-center gap-3 mb-10">
          <div className="w-10 h-10 bg-emerald-500 rounded-lg flex items-center justify-center">
             <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <div>
            <h1 className="font-bold text-lg">BioJarvis</h1>
            <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Agent Monitor</p>
          </div>
        </div>

        <GPUStats 
          used={systemStatus.gpuMemoryUsed} 
          total={systemStatus.gpuMemoryTotal} 
          model={systemStatus.activeModel} 
        />

        <div className="mt-8 space-y-4">
           <div className="bg-slate-800/40 p-3 rounded-lg border border-slate-700">
              <div className="flex justify-between items-center mb-2">
                <span className="text-[10px] text-slate-400 font-bold uppercase">推理引擎</span>
                <span className="text-[10px] text-emerald-400">运行中</span>
              </div>
              <p className="text-xs text-slate-300">Qwen-2.5-7B (4-bit)</p>
           </div>
           <div className="bg-slate-800/40 p-3 rounded-lg border border-slate-700">
              <div className="flex justify-between items-center mb-2">
                <span className="text-[10px] text-slate-400 font-bold uppercase">生物适配器</span>
                <span className="text-[10px] text-blue-400">LoRA Active</span>
              </div>
              <p className="text-xs text-slate-300">ESM-2-650M</p>
           </div>
        </div>

        <div className="mt-auto p-4 bg-slate-950/50 rounded-xl border border-slate-800">
           <p className="text-[10px] text-slate-500 leading-tight">
             VS Code 端口转发状态: <br/>
             <span className="text-emerald-500 font-mono">8000 -> AWS:8000 (OK)</span>
           </p>
        </div>
      </aside>

      <main className="flex-1 flex flex-col bg-[radial-gradient(circle_at_top_right,_var(--tw-gradient-stops))] from-slate-900 via-slate-950 to-slate-950">
        <header className="h-16 border-b border-slate-800 flex items-center px-8 justify-between bg-slate-900/30 backdrop-blur-sm">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isLoading ? 'bg-amber-500 animate-ping' : 'bg-emerald-500'}`} />
            <span className="text-xs font-medium text-slate-400">Agent: {systemStatus.modelStatus}</span>
          </div>
          <button className="text-xs bg-slate-800 hover:bg-slate-700 px-3 py-1.5 rounded-md transition-colors">新建分析任务</button>
        </header>

        <div ref={scrollRef} className="flex-1 overflow-y-auto p-8 space-y-6">
          {messages.map((msg) => (
            <div key={msg.id} className={`flex ${msg.role === MessageRole.USER ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[85%] ${msg.role === MessageRole.USER ? 'bg-emerald-600' : 'bg-slate-800'} p-4 rounded-2xl border border-white/5 shadow-xl`}>
                <p className="text-sm leading-relaxed">{msg.content}</p>
                {msg.metadata?.sequenceData && <ProteinView data={msg.metadata.sequenceData} />}
                {msg.metadata?.processingTime && (
                  <div className="mt-2 pt-2 border-t border-white/10 text-[9px] font-mono text-slate-400 flex justify-between">
                    <span>Engine: L4-NVIDIA</span>
                    <span>Latency: {msg.metadata.processingTime}s</span>
                  </div>
                )}
              </div>
            </div>
          ))}
          {isLoading && <div className="text-xs text-slate-500 animate-pulse">Agent 正在通过 RAG 知识库检索并生成报告...</div>}
        </div>

        <div className="p-8">
          <form onSubmit={handleSubmit} className="relative group">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              className="w-full bg-slate-900 border border-slate-800 rounded-2xl p-4 pr-32 focus:outline-none focus:border-emerald-500/50 transition-all shadow-2xl h-24 text-sm"
              placeholder="输入蛋白质序列进行 LoRA 推理，或直接提问..."
            />
            <button 
              type="submit" 
              className="absolute right-4 bottom-4 bg-emerald-500 hover:bg-emerald-400 text-white px-6 py-2 rounded-xl text-xs font-bold transition-transform active:scale-95 shadow-lg shadow-emerald-500/20"
            >
              发送指令
            </button>
          </form>
        </div>
      </main>
    </div>
  );
};

export default App;
