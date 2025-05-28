import asyncio
import pyaudio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PyQt5.QtCore import QMetaObject, Q_ARG, Qt
from PyQt5 import QtGui

class AudioPlayer:
    def __init__(self, gui_instance):
        self.gui_instance = gui_instance
        self.pyaudio_instance = None
        self.stream = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.current_stream = None
        self.is_paused = False
        self.is_stopped = False
        self._lock = asyncio.Lock()  # 添加锁以保护并发访问

    def _init_pyaudio(self, sample_rate):
        if self.pyaudio_instance is None:
            self.pyaudio_instance = pyaudio.PyAudio()
        if self.stream is None or not self.stream.is_active():
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            self.stream = self.pyaudio_instance.open(format=pyaudio.paFloat32,
                                                     channels=1,
                                                     rate=sample_rate,
                                                     output=True)

    async def stop_current_audio(self):
        """停止当前正在播放的音频"""
        async with self._lock:  # 使用锁来保护资源访问
            self.is_stopped = True  # 首先设置停止标志
            self.is_paused = False  # 重置暂停状态
            
            if self.stream:
                try:
                    if self.stream.is_active():
                        self.stream.stop_stream()
                    self.stream.close()
                except Exception as e:
                    print(f"Error stopping stream: {e}")
                self.stream = None
                
            if self.pyaudio_instance:
                try:
                    self.pyaudio_instance.terminate()
                except Exception as e:
                    print(f"Error terminating PyAudio: {e}")
                self.pyaudio_instance = None

    def _play_blocking(self, audio_array_bytes, sample_rate):
        p = pyaudio.PyAudio()
        self.stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sample_rate, output=True)
        self.stream.write(audio_array_bytes)
        self.stream.stop_stream()
        self.stream.close()
        p.terminate()
        self.stream = None

    async def play_audio_chunk_async(self, audio_array, sample_rate):
        """异步播放音频块"""
        async with self._lock:
            if self.is_stopped:
                return
                
            audio_bytes = audio_array.tobytes() if isinstance(audio_array, np.ndarray) else audio_array
            
            # 初始化 PyAudio 实例和流
            if not self.pyaudio_instance:
                self.pyaudio_instance = pyaudio.PyAudio()
                
            if not self.stream or not self.stream.is_active():
                self.stream = self.pyaudio_instance.open(
                    format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    output=True
                )
                
            try:
                await asyncio.get_event_loop().run_in_executor(
                    self.executor, 
                    self._play_blocking, 
                    audio_bytes, 
                    sample_rate
                )
            except Exception as e:
                print(f"Error playing audio chunk: {e}")
                if not self.is_paused:  # 如果不是暂停导致的错误，需要清理资源
                    await self._cleanup()

    async def run_audio_player_loop(self, audio_queue):
        """运行音频播放循环"""
        try:
            while True:
                if self.is_stopped:
                    print("音频播放已停止")
                    break

                async with self._lock:
                    is_currently_paused = self.is_paused
                
                if is_currently_paused:
                    await asyncio.sleep(0.1)
                    continue

                try:
                    # 使用timeout来避免永久阻塞
                    item = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break

                if item is None:
                    print("音频队列中收到停止信号")
                    break

                audio_data, sample_rate, chunk = item
                
                try:
                    # 更新UI显示当前朗读内容
                    QMetaObject.invokeMethod(
                        self.gui_instance,
                        "highlight_paragraph",
                        Qt.QueuedConnection,
                        Q_ARG(str, chunk["text"]),
                        Q_ARG(int, chunk["page"]),
                    )

                    # 再次检查是否暂停，避免在UI更新后播放
                    async with self._lock:
                        if self.is_paused or self.is_stopped:
                            await audio_queue.put(item)  # 放回未播放的内容
                            continue

                    # 异步播放音频
                    await self.play_audio_chunk_async(audio_data, sample_rate)
                    
                    # 完成后标记任务完成
                    audio_queue.task_done()
                    
                except Exception as e:
                    print(f"处理音频块时出错: {e}")
                    if not self.is_paused and not self.is_stopped:
                        audio_queue.task_done()

        except Exception as e:
            print(f"音频播放循环出错: {e}")
        finally:
            print("音频播放循环结束，清理资源")
            await self._cleanup()  # 使用异步清理方法

    async def pause(self):
        """暂停播放"""
        print("暂停音频播放")
        async with self._lock:  # 使用锁来保护状态更改
            self.is_paused = True
            if not self.is_stopped:  # 只有在未停止时才尝试暂停流
                current_stream = self.stream 
                if current_stream:
                    try:
                        if current_stream.is_active():
                            current_stream.stop_stream()
                            print("Stream paused successfully.")
                    except Exception as e:
                        print(f"Error pausing stream: {e}")

    async def resume(self):
        """恢复播放"""
        print("恢复音频播放")
        async with self._lock:  # 使用锁来保护状态更改
            if not self.is_stopped:  # 只有在未停止时才恢复
                self.is_paused = False
                # 如果有暂停的流，尝试重新启动
                current_stream = self.stream
                if current_stream and not current_stream.is_active():
                    try:
                        current_stream.start_stream()
                        print("Stream resumed successfully.")
                    except Exception as e:
                        print(f"Error resuming stream: {e}")

    async def _cleanup(self):
        """清理音频资源"""
        async with self._lock:  # 使用锁来保护资源访问
            if self.stream:
                try:
                    if self.stream.is_active():
                        self.stream.stop_stream()
                    self.stream.close()
                except Exception as e:
                    print(f"Error stopping stream during cleanup: {e}")
                self.stream = None
                
            if self.pyaudio_instance:
                try:
                    self.pyaudio_instance.terminate()
                except Exception as e:
                    print(f"Error terminating PyAudio during cleanup: {e}")
                self.pyaudio_instance = None
                
            self.is_stopped = True  # 确保设置停止状态
            print("stop 1")
            self.is_paused = False  # 重置暂停状态
            
            try:
                if not self.executor._shutdown:
                    self.executor.shutdown(wait=False)
            except Exception as e:
                print(f"Error shutting down executor: {e}")

    def update_current_page_display(self, text, page_num):
        self.text_display.setText(text)
        self.currently_displayed_page_num_0_indexed = page_num

    def highlight_paragraph(self, text, page_num):
        # 如果需要切换页面
        if page_num != self.currently_displayed_page_num_0_indexed:
            self.display_page(page_num)
        
        # 高亮当前朗读的文本
        cursor = self.text_display.textCursor()
        cursor.setPosition(0)
        found_pos = self.text_display.toPlainText().find(text)
        if found_pos >= 0:
            cursor.setPosition(found_pos)
            cursor.setPosition(found_pos + len(text), QtGui.QTextCursor.KeepAnchor)
            format = QtGui.QTextCharFormat()
            format.setBackground(QtGui.QColor("yellow"))
            cursor.mergeCharFormat(format)