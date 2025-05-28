import sys
import os
import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import fitz
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout, QHBoxLayout, 
    QWidget, QLabel, QSpinBox, QFileDialog, QTreeView, QSplitter, QSizePolicy,
    QLineEdit, QDoubleSpinBox, QGroupBox, QMessageBox
)
from PyQt5.QtCore import Qt, QMetaObject, Q_ARG, QTimer, pyqtSlot
from PyQt5.QtGui import QBrush, QColor, QFont, QTextCharFormat, QTextCursor, QStandardItemModel, QStandardItem
from pdf_utils import extract_text_chunks_from_range, get_pdf_outline
from tts_utils import load_tts_resources, generate_audio_chunk, clean_text_for_tts
from audio_player_utils import AudioPlayer

# 获取资源目录的基路径
if getattr(sys, 'frozen', False):
    # 如果是 PyInstaller 打包的，使用 _MEIPASS
    # base_path = sys._MEIPASS
    base_path = os.path.dirname(sys.executable) # .exe 所在的目录
else:
    # 否则，使用脚本所在的目录
    base_path = os.path.dirname(os.path.abspath(__file__))
    base_path =os.path.join(base_path, '..')

# 默认设置
DEFAULT_TTS_MODEL_PATH = os.path.join(base_path, 'resources', 'models', 'F5TTS_v1_Base', 'model_1250000.safetensors')
DEFAULT_TTS_VOCAB_PATH = os.path.join(base_path, 'resources', 'models', 'F5TTS_v1_Base', 'vocab.txt')
DEFAULT_REF_AUDIO_PATH = os.path.join(base_path, 'resources', 'ref_voice', '读书男声2.WAV')
DEFAULT_SPEED = 1.0
DEFAULT_LAST_PDF_PATH = ""
DEFAULT_LAST_READ_PAGE = 0
SETTINGS_FILE = os.path.join(base_path, 'data', 'autoreader_settings.json')
TEXT_CHUNK_LENGTH = 50

class AutoReaderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoReader 有声阅读器")
        self.setGeometry(200, 200, 1000, 700)

        # 初始化状态变量
        self.pdf_path = None
        self.pdf_doc = None
        self.pdf_outline = []
        self.current_chunk = ""
        self.is_reading = False
        self.is_reading_paused = False
        self.currently_displayed_page_num_0_indexed = -1
        self.stop_reading_flag = False # 用于停止朗读的布尔标志
        self.shutdown_timer = QTimer(self)
        self.audio_player = None
        self.audio_queue = asyncio.Queue()

        # 添加事件循环和线程相关的属性
        self.loop = None
        self.thread = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.tts_resources_loaded = False

        # 加载设置
        self.load_settings()

        # 定义标准按钮样式，以便在UI初始化中使用
        self.standard_button_style = """
            QPushButton {
                background-color: #f0f0f0; /* 按钮基色 */
                border: 1px solid #cccccc;
                border-radius: 6px; /* 统一圆角为6px */
                padding: 3px 7px; /* 适用于非固定大小的按钮，固定大小按钮主要由其尺寸决定 */
            }
            QPushButton:hover { background-color: #e0e0e0; }
            QPushButton:pressed { background-color: #d0d0d0; }
            QPushButton:disabled { background-color: #e8e8e8; color: #a0a0a0; border-color: #d8d8d8; } /* 禁用状态 */
        """
        
        # 初始化异步运行环境
        self._init_async_environment()

        # 初始化UI
        self.init_ui()
        self._auto_load_last_session() # 自动加载上次会话
        
    def _init_async_environment(self):
        """初始化异步运行环境"""
        def run_event_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()
            
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=run_event_loop, args=(self.loop,), daemon=True)
        self.thread.start()
        
    def init_ui(self):
        # 主布局
        main_widget = QWidget()
        main_widget.setStyleSheet("QWidget { background-color: #FAFAFA; }") # 设置主背景色
        self.setCentralWidget(main_widget)
        self.main_v_layout = QVBoxLayout(main_widget) # Renamed for clarity
        self.main_v_layout.setSpacing(4)  # 优化主要UI块之间的垂直间距
        self.main_v_layout.setContentsMargins(8, 8, 8, 8)  # 优化整体边距

        # 顶部区域：设置按钮和文件选择
        top_layout = QHBoxLayout()
        top_layout.setSpacing(5) # 优化顶部区域内部间距
        
        # 设置按钮
        settings_button = QPushButton("设置")
        settings_button.setFixedSize(80, 30)  # 细长边框
        settings_button.clicked.connect(self.toggle_settings_visibility)
        settings_button.setStyleSheet(self.standard_button_style)
        
        # 文件选择按钮
        self.load_button = QPushButton("打开 PDF")
        self.load_button.setFixedSize(80, 30)
        self.load_button.clicked.connect(self.load_pdf) # 连接“打开PDF”按钮到load_pdf方法
        self.load_button.setStyleSheet(self.standard_button_style)
        
        # 文件路径标签
        self.file_path_label = QLabel("未选择文件")
        self.file_path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred) # 允许标签横向扩展
        
        top_layout.addWidget(settings_button)
        top_layout.addSpacing(5) # 调整按钮间间距
        top_layout.addWidget(self.load_button)
        top_layout.addWidget(self.file_path_label, 1) # 给文件路径标签一个拉伸因子
        
        self.main_v_layout.addLayout(top_layout) # 添加到主布局，默认不拉伸
        
        # 设置区域
        self.settings_group = QGroupBox("TTS 和音频设置")
        self.settings_group.setVisible(False)
        settings_layout = self.create_settings_section()
        self.settings_group.setLayout(settings_layout)
        self.main_v_layout.addWidget(self.settings_group) # 添加到主布局，默认不拉伸
        
        # 主内容区域（左右双栏）
        content_splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：目录树
        self.outline_view = QTreeView()
        self.outline_view.setStyleSheet("""
            QTreeView {
                background-color: #FAFAFA; /* 与主背景色一致或可根据需要微调 */
                /* 字体将由全局字体设置控制 */
                border: none;
            }
            QTreeView::item {
                height: 24px;
            }
        """)
        content_splitter.addWidget(self.outline_view)
        
        # 右侧：文本显示
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setStyleSheet("""
            QTextEdit {
                background-color: white;
                color: #333333; /* 深灰色文本，保持不变 */
                /* 字体将由全局字体设置控制 */
                font-size: 16px;
                line-height: 1.5;
                padding: 10px;
            }
        """)
        content_splitter.addWidget(self.text_display)
        
        # 设置分割比例
        content_splitter.setStretchFactor(0, 1)  # 目录区域
        content_splitter.setStretchFactor(1, 3)  # 内容区域
        self.main_v_layout.addWidget(content_splitter, 1) # 添加到主布局，并设置拉伸因子为1，使其填充可用空间
        
        # 底部控制区域
        control_layout = QHBoxLayout()
        control_layout.setSpacing(6) # 优化底部控制区域内部间距
        
        self.start_button = QPushButton("开始朗读")
        self.start_button.setFixedSize(100, 30)
        self.pause_button = QPushButton("暂停")
        self.pause_button.setFixedSize(100, 30)
        self.pause_button.setEnabled(False)
        
        self.page_info_label = QLabel("第 0 / 0 页")    # 在init_ui方法底部控制区域添加页码标签
        self.page_info_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.page_info_label.setStyleSheet("color: #666666; font-size: 14px;")

        self.prev_page_button = QPushButton("上一页")
        self.prev_page_button.setFixedSize(80, 30)
        self.prev_page_button.setStyleSheet(self.standard_button_style)
        self.prev_page_button.clicked.connect(self.goto_prev_page)

        self.next_page_button = QPushButton("下一页")
        self.next_page_button.setFixedSize(80, 30)
        self.next_page_button.setStyleSheet(self.standard_button_style)
        self.next_page_button.clicked.connect(self.goto_next_page)

        # 设置按钮样式
        self.start_button.setStyleSheet(self.standard_button_style)
        self.pause_button.setStyleSheet(self.standard_button_style)
        
        # Modify the connection for the start button
        self.start_button.clicked.connect(self.on_start_reading_clicked)
        self.pause_button.clicked.connect(self.pause_resume_reading)
        
        control_layout.addWidget(self.page_info_label, 0)  # 添加到底部控制栏最左侧
        control_layout.addStretch()
        control_layout.addWidget(self.start_button)
        control_layout.addSpacing(5) # 按钮间距
        control_layout.addWidget(self.pause_button)
        control_layout.addStretch()
        control_layout.addWidget(self.prev_page_button)
        control_layout.addSpacing(5)
        control_layout.addWidget(self.next_page_button)
        
        self.main_v_layout.addLayout(control_layout) # 添加到主布局，默认不拉伸

    def create_settings_section(self):
        settings_layout = QVBoxLayout()
        
        # 模型路径设置
        model_layout = QHBoxLayout()
        model_label = QLabel("模型路径:")
        self.model_path_input = QLineEdit(self.tts_model_path)
        model_browse = QPushButton("浏览")
        model_browse.clicked.connect(lambda: self.browse_file(self.model_path_input))
        model_browse.setStyleSheet(self.standard_button_style)
        model_layout.setSpacing(5)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_path_input)
        model_layout.addWidget(model_browse)
        settings_layout.addLayout(model_layout)
        
        # 词汇表路径设置
        vocab_layout = QHBoxLayout()
        vocab_label = QLabel("词汇表路径:")
        self.vocab_path_input = QLineEdit(self.tts_vocab_path)
        vocab_browse = QPushButton("浏览")
        vocab_browse.clicked.connect(lambda: self.browse_file(self.vocab_path_input))
        vocab_browse.setStyleSheet(self.standard_button_style)
        vocab_layout.setSpacing(5)
        vocab_layout.addWidget(vocab_label)
        vocab_layout.addWidget(self.vocab_path_input)
        vocab_layout.addWidget(vocab_browse)
        settings_layout.addLayout(vocab_layout)
        
        # 参考音频路径设置
        ref_audio_layout = QHBoxLayout()
        ref_audio_label = QLabel("参考音频路径:")
        self.ref_audio_path_input = QLineEdit(self.ref_audio_path)
        ref_audio_browse = QPushButton("浏览")
        ref_audio_browse.clicked.connect(lambda: self.browse_file(self.ref_audio_path_input))
        ref_audio_browse.setStyleSheet(self.standard_button_style)
        ref_audio_layout.setSpacing(5)
        ref_audio_layout.addWidget(ref_audio_label)
        ref_audio_layout.addWidget(self.ref_audio_path_input)
        ref_audio_layout.addWidget(ref_audio_browse)
        settings_layout.addLayout(ref_audio_layout)
        
        # 语速设置
        speed_layout = QHBoxLayout()
        speed_label = QLabel("语速:")
        self.speed_input = QDoubleSpinBox()
        self.speed_input.setRange(0.1, 5.0)
        self.speed_input.setSingleStep(0.1)
        self.speed_input.setValue(self.tts_speed)
        speed_layout.addWidget(speed_label)
        speed_layout.setSpacing(5)
        speed_layout.addWidget(self.speed_input)
        speed_layout.addStretch()
        settings_layout.addLayout(speed_layout)
        
        # 定时关机设置
        shutdown_layout = QHBoxLayout()
        self.shutdown_spinbox = QSpinBox()
        self.shutdown_spinbox.setRange(0, 1440)  # 最多24小时
        self.shutdown_spinbox.setSuffix(" 分钟")
        self.shutdown_button = QPushButton("定时关机")
        self.shutdown_button.clicked.connect(self.start_shutdown_timer)
        self.shutdown_button.setStyleSheet(self.standard_button_style)
        self.cancel_shutdown_button = QPushButton("取消定时")
        self.cancel_shutdown_button.clicked.connect(self.cancel_shutdown_timer)
        self.cancel_shutdown_button.setStyleSheet(self.standard_button_style)
        shutdown_layout.setSpacing(5)
        shutdown_layout.addWidget(self.shutdown_spinbox)
        shutdown_layout.addWidget(self.shutdown_button)
        shutdown_layout.addWidget(self.cancel_shutdown_button)
        shutdown_layout.addStretch()
        settings_layout.addLayout(shutdown_layout)
        
        # 修改保存设置按钮的文本
        save_button = QPushButton("保存设置并重新加载TTS资源")
        save_button.clicked.connect(self.save_and_reload)
        save_button.setStyleSheet(self.standard_button_style)
        settings_layout.addWidget(save_button)
        
        settings_layout.addStretch()
        return settings_layout

    def browse_file(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "所有文件 (*.*)")
        if file_path:
            line_edit.setText(file_path)

    def toggle_settings_visibility(self):
        self.settings_group.setVisible(not self.settings_group.isVisible())

    def start_shutdown_timer(self):
        minutes = self.shutdown_spinbox.value()
        if minutes > 0:
            self.shutdown_timer.stop()  # 停止之前的计时器
            self.shutdown_timer.singleShot(minutes * 60 * 1000, self.execute_shutdown)
            QMessageBox.information(self, "定时关机", f"系统将在 {minutes} 分钟后关机")

    def cancel_shutdown_timer(self):
        self.shutdown_timer.stop()
        QMessageBox.information(self, "取消定时关机", "已取消定时关机")

    def execute_shutdown(self):
        # 保存当前状态
        self.save_settings()
        # 执行关机命令
        os.system("shutdown /s /t 60")  # 60秒后关机
        QMessageBox.information(self, "关机提示", "系统将在1分钟后关机")

    def _auto_load_last_session(self):
        """应用启动时自动加载上一次的PDF和页码"""
        # 尝试从历史记录中获取上次阅读的页码
        last_page = self.pdf_page_history.get(self.last_pdf_path, DEFAULT_LAST_READ_PAGE)
        if self.last_pdf_path and os.path.exists(self.last_pdf_path):
            try:
                print(f"尝试自动加载: {self.last_pdf_path} 第 {last_page} 页")
                self.pdf_path = self.last_pdf_path 
                self.file_path_label.setText(self.last_pdf_path)
                # 调用 load_pdf_content 来加载PDF内容和目录
                self.load_pdf_content(self.last_pdf_path, initial_page=last_page) # 传递上次阅读的页码 
                print(f"加载PDF成功: {self.pdf_path}")

                # if self.pdf_doc: 
                #     if 0 <= self.last_read_page < self.pdf_doc.page_count:
                #         self.display_page(self.last_read_page)
                #     else:
                #         print(f"记录的页码 {self.last_read_page + 1} 超出范围。显示第一页。")
                #         self.display_page(0) 
            except Exception as e:
                print(f"自动加载上次会话失败{last_page}: {e}")
                self.file_path_label.setText("自动加载上次文件失败")
                self.pdf_path = None 

    def load_settings(self):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings = json.load(f)
                self.tts_model_path = settings.get("tts_model_path", DEFAULT_TTS_MODEL_PATH)
                self.tts_vocab_path = settings.get("vocab_path", DEFAULT_TTS_VOCAB_PATH) # 确保键名一致
                self.ref_audio_path = settings.get("ref_audio_path", DEFAULT_REF_AUDIO_PATH)
                self.tts_speed = settings.get("speed", DEFAULT_SPEED)
                # 加载PDF页码历史字典
                self.pdf_page_history = settings.get("pdf_page_history", {}) # 新增：加载页码历史字典
                # 保留 last_pdf_path 用于应用启动时自动加载，其页码将从 pdf_page_history 中获取
                self.last_pdf_path = settings.get("last_pdf_path", DEFAULT_LAST_PDF_PATH)

        except FileNotFoundError:
            print(f"Settings file '{SETTINGS_FILE}' not found. Using default settings.")
            # 使用默认值
            self.tts_model_path = DEFAULT_TTS_MODEL_PATH
            self.tts_vocab_path = DEFAULT_TTS_VOCAB_PATH
            self.ref_audio_path = DEFAULT_REF_AUDIO_PATH
            self.tts_speed = DEFAULT_SPEED
            self.pdf_page_history = {} # 新增：初始化为空字典
            self.last_pdf_path = DEFAULT_LAST_PDF_PATH

        except Exception as e:
            print(f"Error loading settings from '{SETTINGS_FILE}': {e}. Using default settings.")
            # 其他错误（如JSON解析错误），也使用默认值
            self.tts_model_path = DEFAULT_TTS_MODEL_PATH
            self.tts_vocab_path = DEFAULT_TTS_VOCAB_PATH
            self.ref_audio_path = DEFAULT_REF_AUDIO_PATH
            self.tts_speed = DEFAULT_SPEED
            self.pdf_page_history = {} # 新增：初始化为空字典
            self.last_pdf_path = DEFAULT_LAST_PDF_PATH

    def save_settings(self):
        """保存设置,包括TTS设置和PDF页码历史"""
        try:
            # 读取现有设置，以便更新页码历史而不是完全覆盖
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    settings = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                settings = {}

            # 更新TTS设置
            settings["tts_model_path"] = self.model_path_input.text()
            settings["vocab_path"] = self.vocab_path_input.text()
            settings["ref_audio_path"] = self.ref_audio_path_input.text()
            settings["speed"] = self.speed_input.value()

            # 更新当前PDF的页码历史
            if self.pdf_path:
                # 确保 pdf_page_history 存在且是字典
                if "pdf_page_history" not in settings or not isinstance(settings["pdf_page_history"], dict):
                    settings["pdf_page_history"] = {}
                settings["pdf_page_history"][self.pdf_path] = self.currently_displayed_page_num_0_indexed
                # 同时更新 last_pdf_path 为当前文件路径
                settings["last_pdf_path"] = self.pdf_path
            else:  # 如果没有打开的PDF，则不更新页码历史和last_pdf_path
                pass # 或者可以选择清空 last_pdf_path: settings["last_pdf_path"] = DEFAULT_LAST_PDF_PATH

            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)

            # 更新内部状态 (可选，因为下次加载会重新读取)
            # self.tts_model_path = settings["tts_model_path"]
            # self.tts_vocab_path = settings["vocab_path"]
            # self.ref_audio_path = settings["ref_audio_path"]
            # self.tts_speed = settings["speed"]
            # self.pdf_page_history = settings["pdf_page_history"]
            # self.last_pdf_path = settings["last_pdf_path"]

            return True

        except Exception as e:
            QMessageBox.critical(self, "保存设置错误", f"保存设置时发生错误: {str(e)}")
            return False

    def reload_tts_resources(self):
        """重新加载TTS资源"""
        try:
            success = load_tts_resources(
                self.model_path_input.text(),
                self.vocab_path_input.text(),
                None,
                self.ref_audio_path_input.text(),
                "",
                None
            )
            self.tts_resources_loaded = success
            
            if not success:
                QMessageBox.warning(self, "加载警告", "TTS资源加载失败,请检查设置和文件路径")
                
            return success
            
        except Exception as e:
            QMessageBox.critical(self, "加载错误", f"加载TTS资源时发生错误: {str(e)}")
            return False

    def save_and_reload(self):
        """保存设置并重新加载TTS资源"""
        if self.save_settings():
            self.reload_tts_resources()

    def load_pdf(self):
        """打开新的PDF文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择PDF文件", "", "PDF文件 (*.pdf)")
        if file_path:
            # 保存当前PDF的页码历史
            self.save_settings()
            # 退出朗读
            self.stop_reading_flag = True
            self.pdf_path = file_path
            # 尝试从历史记录中获取上次阅读的页码
            last_page = self.pdf_page_history.get(file_path, DEFAULT_LAST_READ_PAGE)
            print(f"路径：{file_path}，历史页码：{last_page}")
            self.file_path_label.setText(file_path)
            # 调用 load_pdf_content 来加载PDF内容和目录
            self.load_pdf_content(file_path, initial_page=last_page) # 传递上次阅读的页码

            # self.load_pdf_content(file_path)

    async def cleanup_resources(self):
        """清理异步资源"""
        try:
            # 停止当前音频播放
            if self.audio_player:
                await self.audio_player.stop_current_audio()
                # 清空音频队列
                while not self.audio_queue.empty():
                    print("清空音频队列")
                    try:
                        self.audio_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                self.audio_player = None

            # 重置状态
            self.stop_reading_flag = True
            self.is_reading_paused = False
            self.pause_button.setText("暂停")
            self.pause_button.setEnabled(False)
            
            # 清除高亮
            self.clear_all_highlights()
            
            # 等待所有任务完成
            await asyncio.sleep(0.1)
            
        except Exception as e:
            print(f"清理资源时出错: {e}")

    def load_pdf_content(self, path, initial_page=0):
        try:
            self.pdf_doc = fitz.open(path)
            self.pdf_outline = get_pdf_outline(self.pdf_doc)
            self.update_outline_view()
            # 确定要显示的起始页码
            page_to_display = initial_page
            # 如果指定的起始页码无效，则显示第一页
            if not (0 <= page_to_display < self.pdf_doc.page_count):
                 page_to_display = 0

            print(f"加载文件: {path}. 显示页码: {page_to_display + 1}")
            self.display_page(page_to_display)
            
            # 加载PDF后立即刷新页码显示
            self.page_info_label.setText(f"第 {page_to_display + 1} / {self.pdf_doc.page_count} 页")
            
        except Exception as e:
            error_msg = f"无法加载PDF文件: {str(e)}"
            self.text_display.setText(error_msg)
            QMessageBox.critical(self, "错误", error_msg)

    def display_page(self, page_num):
        if self.pdf_doc and page_num < self.pdf_doc.page_count:
            page = self.pdf_doc.load_page(page_num)
            text = page.get_text("text")
            self.clear_all_highlights()
            self.text_display.setText(text)
            self.currently_displayed_page_num_0_indexed = page_num
            
            # 更新页码显示
            self.page_info_label.setText(f"第 {page_num + 1} / {self.pdf_doc.page_count} 页")

            # 更新目录视图中项目的背景颜色
            self._highlight_current_outline_item(page_num)

    def _highlight_current_outline_item(self, current_page_num):
        """根据当前页码高亮目录视图中对应的章节项"""
        print(f"_highlight_current_outline_item called for page: {current_page_num}") # Debug print
        model = self.outline_view.model()
        if not model:
            print("No outline model found.") # Debug print
            return

        # 1. 查找与当前页码最匹配的目录项（页码 <= 当前页码，且页码最大）
        best_match_item = None
        best_match_page = -1 # 0-indexed page number

        # 递归查找最佳匹配项
        def find_best_match(parent_item, current_best_item, current_best_page):
            if isinstance(parent_item, QStandardItemModel):
                rows = parent_item.rowCount()
                get_child = lambda r: parent_item.item(r)
            else:
                rows = parent_item.rowCount()
                get_child = lambda r: parent_item.child(r)

            for row in range(rows):
                item = get_child(row)
                if not item:
                    continue

                page_data = item.data(Qt.UserRole)
                if page_data is not None:
                    item_page_num_0_indexed = page_data - 1
                    if item_page_num_0_indexed >= 0 and item_page_num_0_indexed <= current_page_num:
                        if item_page_num_0_indexed > current_best_page:
                            current_best_page = item_page_num_0_indexed
                            current_best_item = item

                # 递归搜索子项
                current_best_item, current_best_page = find_best_match(item, current_best_item, current_best_page)

            return current_best_item, current_best_page

        best_match_item, best_match_page = find_best_match(model, best_match_item, best_match_page)

        if best_match_item:
            print(f"Best match found: '{best_match_item.text()}' on page {best_match_page + 1}") # Debug print
        else:
            print("No best match item found.") # Debug print

        # 2. 遍历所有目录项，只高亮最佳匹配项
        # 使用非递归方式遍历所有item并设置颜色
        items_to_process = [model.item(r) for r in range(model.rowCount())]
        while items_to_process:
            item = items_to_process.pop(0)
            if not item:
                continue

            item_page_data = item.data(Qt.UserRole)
            item_page_str = f" (Page: {item_page_data})" if item_page_data is not None else ""
            # print(f"Processing item: '{item.text()}'{item_page_str}") # Debug print

            if item is best_match_item:
                # print(f"  -> Setting '{item.text()}' to GREY") # Debug print
                item.setBackground(QBrush(QColor("#D3D3D3")))
            else:
                # print(f"  -> Setting '{item.text()}' to TRANSPARENT") # Debug print
                item.setBackground(QBrush(Qt.transparent))

            # 将子项添加到待处理列表
            for r in range(item.rowCount()):
                child_item = item.child(r)
                if child_item:
                    items_to_process.append(child_item)

        # apply_colors(model, best_match_item) # 移除递归调用

    def _clear_outline_highlights(self, parent_item):
        """递归清除所有目录项的背景高亮"""
        if isinstance(parent_item, QStandardItemModel):
            rows = parent_item.rowCount()
            get_child = lambda r: parent_item.item(r)
        else:
            rows = parent_item.rowCount()
            get_child = lambda r: parent_item.child(r)

        for row in range(rows):
            item = get_child(row)
            if item:
                item.setBackground(QBrush(Qt.transparent))
                # 递归清除子项高亮
                self._clear_outline_highlights(item)

    def _update_outline_item_color(self, item, current_page_num):
        # 递归更新子项目的颜色
        for row in range(item.rowCount()):
            child_item = item.child(row)
            if child_item:
                self._update_outline_item_color(child_item, current_page_num)

        # 更新当前项目的颜色
        page_data = item.data(Qt.UserRole)
        if page_data is not None:
            item_page_num_0_indexed = page_data - 1
            if item_page_num_0_indexed >= 0:
                if item_page_num_0_indexed <= current_page_num:
                    # 设置背景颜色为灰色
                    item.setBackground(QBrush(QColor("#D3D3D3")))
                else:
                    # 设置背景颜色为透明
                    item.setBackground(QBrush(Qt.transparent))

    def on_start_reading_clicked(self):
        """Handle start button click to start reading in the correct loop."""
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.start_reading(), self.loop)
        else:
            print("Event loop not running, cannot start reading.")

    async def start_reading(self, start_page=None, end_page=None):
        print("开始朗读按钮点击")
        print(f"起始页: {start_page}, 结束页: {end_page}")
        print(f"self.is_reading: {self.is_reading}")

        if self.is_reading:
            print("已有朗读任务，等待其退出...")
            self.stop_reading_flag = True
            # 等待主循环退出
            while self.is_reading:
                await asyncio.sleep(0.1)
            print("上一个朗读任务已退出")

        if not self.pdf_doc:
            QMessageBox.warning(self, "警告", "请先加载一个PDF文件")
            return

        self.is_reading = True
        print(f"=========> start_reading is_reading:{self.is_reading}")
        self.stop_reading_flag = False
        self.is_reading_paused = False
        self.start_button.setDisabled(True)
        self.pause_button.setDisabled(False)

        print("音频队列已清空并重新创建")

        current_page = start_page if start_page is not None else self.currently_displayed_page_num_0_indexed
        _end_page = end_page if end_page is not None else self.pdf_doc.page_count -1

        if current_page is None or current_page < 0:
            current_page = 0

        print(f"实际朗读起始页: {current_page}, 结束页: {_end_page}")
        asyncio.create_task(
            self.main_async_reader_loop(current_page, _end_page)
        )
        print(f"main_async_reader_loop 已调度")


    async def main_async_reader_loop(self, start_page, _end_page): # _end_page 未使用，改为读取整个文档
        """主异步朗读循环"""
        print(f"开始新的朗读循环, stop_reading_flag 状态: {self.stop_reading_flag}") # 调试打印 (布尔值)
        try:
            # 确保TTS资源已加载
            if not self.tts_resources_loaded:
                if not self.reload_tts_resources():
                    return

            # 启动音频播放器
            if not self.audio_player:
                self.audio_player = AudioPlayer(self)
                asyncio.create_task(
                    self.audio_player.run_audio_player_loop(self.audio_queue)
                )

            # 设置最大队列大小
            MAX_QUEUE_SIZE = 2

            # 异步生成并处理文本块
            async for chunk in extract_text_chunks_from_range(
                self.pdf_doc,
                start_page,
                self.pdf_doc.page_count - 1, # 朗读到文档末尾
                chunk_length=50,
                stop_reading_flag=self.stop_reading_flag,
                text_cleaner_func=clean_text_for_tts
            ):
                if self.stop_reading_flag: # 检查停止标志 (布尔值)
                    print("stop_reading_flag set, 朗读循环提前结束")
                    break
                print(f"=====主异步朗读循环=====")
                while self.is_reading_paused and not self.stop_reading_flag:
                    self.start_button.setEnabled(True) # 当朗读暂停时，重新启用开始朗读按钮
                    await asyncio.sleep(0.1)

                # 等待队列大小降到限制以下
                while self.audio_queue.qsize() >= MAX_QUEUE_SIZE and not self.stop_reading_flag:
                    if self.is_reading_paused:
                        self.start_button.setEnabled(True) # 当朗读暂停时，重新启用开始朗读按钮
                    await asyncio.sleep(0.1)

                if self.stop_reading_flag:
                    print("Reading stopped by pause")
                    break # 如果停止标志被设置，退出循环
                    
                try:
                    # 异步生成音频
                    print("准备生成音频")
                    audio_data, sample_rate = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        generate_audio_chunk,
                        chunk["tts_text"],
                        self.speed_input.value()
                    )
                    print("生成音频完成")
                    # 放入队列等待播放
                    await self.audio_queue.put((audio_data, sample_rate, chunk))
                    print(f"当前队列大小: {self.audio_queue.qsize()}")
                    
                except Exception as e:
                    print(f"生成音频时出错: {e}")
                    continue
                
            #等待资源清理完成
            await self.cleanup_resources()
            print(f"=====主异步朗读循环退出=====")

        except Exception as e:
            print(f"朗读循环出错: {e}")
        finally:
            self.is_reading = False
            print(f"=========> 朗读循环退出 is_reading:{self.is_reading}")
            self.start_button.setEnabled(True) # 重新启用开始朗读按钮
            self.pause_button.setEnabled(False) # 禁用暂停按钮
            self.pause_button.setText("暂停")

    def stop_reading(self):
        """停止朗读并清理资源"""
        self.stop_reading_flag = True  # 设置停止标志为 True
        self.clear_all_highlights()
        self.is_reading_paused = False
        self.pause_button.setText("暂停")
        self.pause_button.setEnabled(False)
        self.start_button.setEnabled(True) # 重新启用开始朗读按钮
        
        # 停止音频播放（异步）
        if self.audio_player:
            print("stop vodio 1")
            future = asyncio.run_coroutine_threadsafe(
                self.audio_player.stop_current_audio(),
                self.loop
            )
            print("stop vodio reading")
            try:
                # 等待异步操作完成，但设置超时
                future.result(timeout=1.0)
            except Exception as e:
                print(f"Warning: Failed to stop audio player: {e}")

    def pause_resume_reading(self):
        """暂停或继续朗读"""
        self.is_reading_paused = not self.is_reading_paused
        self.pause_button.setText("继续" if self.is_reading_paused else "暂停")

        # 暂停/恢复音频播放（异步）
        if self.audio_player:
            if self.is_reading_paused:
                asyncio.run_coroutine_threadsafe(
                    self.audio_player.pause(),
                    self.loop
                )
            else:
                asyncio.run_coroutine_threadsafe(
                    self.audio_player.resume(),
                    self.loop
                )

    def goto_prev_page(self):
        if self.pdf_doc and self.currently_displayed_page_num_0_indexed > 0:
            self.display_page(self.currently_displayed_page_num_0_indexed - 1)

    def goto_next_page(self):
        if self.pdf_doc and self.currently_displayed_page_num_0_indexed < self.pdf_doc.page_count - 1:
            self.display_page(self.currently_displayed_page_num_0_indexed + 1)

    def update_outline_view(self):
        # 创建大纲模型并更新视图
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(["目录"])
        
        # 用于跟踪每个级别的最后一个项目
        last_items = {}
        
        for item in self.pdf_outline:
            level = item['level']
            # 在display_title后面添加页码信息
            display_text = item['display_title']
            if item['page'] is not None:
                display_text += f"  ({item['page']})"
            outline_item = QStandardItem(display_text)
            outline_item.setData(item['page'], Qt.UserRole)
            
            # 确定父项
            if level == 0:
                model.appendRow(outline_item)
            else:
                parent_level = level - 1
                if parent_level in last_items:
                    last_items[parent_level].appendRow(outline_item)
                else:
                    # 如果没有找到父项，就添加到根级别
                    model.appendRow(outline_item)
            
            last_items[level] = outline_item
        
        self.outline_view.setModel(model)
        self.outline_view.clicked.connect(self.outline_item_clicked)
        self.outline_view.expandAll()  # 展开所有节点

    def outline_item_clicked(self, index):
        page_data = index.data(Qt.UserRole)
        if page_data is not None:
            page_num_0_indexed = page_data - 1  # 转换为0基索引
            if page_num_0_indexed >= 0:
                self.display_page(page_num_0_indexed)

    @pyqtSlot(str, int)
    def highlight_paragraph(self, text, page_num):
        """高亮显示当前朗读段落并自动滚动到该位置"""
        self.clear_all_highlights()
        # 如果需要切换页面
        page_num_0_indexed = page_num - 1
        if page_num_0_indexed != self.currently_displayed_page_num_0_indexed:
            self.display_page(page_num_0_indexed)
        
        # 获取当前页面的完整文本
        current_text = self.text_display.toPlainText()
        
        # 清理和规范化文本以便更好地匹配
        def normalize_text(t):
            # 移除多余的空白字符
            return ' '.join(t.split())
        
        text_to_find = normalize_text(text)
        current_page_text = normalize_text(current_text)
        
        # 在当前页面文本中查找目标文本的位置
        start_pos = current_page_text.find(text_to_find)
        
        if start_pos >= 0:
            # 创建光标并设置选择区域
            cursor = self.text_display.textCursor()
            cursor.setPosition(start_pos)
            cursor.setPosition(start_pos + len(text_to_find), QTextCursor.KeepAnchor)
            
            # 应用高亮格式
            fmt = QTextCharFormat()
            fmt.setBackground(QColor("#FFF4D6"))  # 柔和的浅黄色
            cursor.mergeCharFormat(fmt)
            
            # 更新光标位置以确保可见
            self.text_display.setTextCursor(cursor)
            self.text_display.ensureCursorVisible()
            
            # 清除选中状态但保持高亮
            cursor.clearSelection()
            self.text_display.setTextCursor(cursor)
            
            print(f"成功高亮文本: '{text_to_find[:30]}...'")
        else:
            print(f"高亮提示: 文本未在当前页面找到: '{text_to_find[:30]}...'")
            print(f"页面文本前100个字符: '{current_page_text[:100]}...'")

    def clear_all_highlights(self):
        """清除文本显示区域的所有背景高亮"""
        cursor = self.text_display.textCursor()
        cursor.select(QTextCursor.Document) # 选中整个文档
        fmt = QTextCharFormat()
        fmt.setBackground(Qt.transparent) # 设置透明背景
        cursor.mergeCharFormat(fmt) # 应用格式
        cursor.clearSelection() # 清除选中状态
        self.text_display.setTextCursor(cursor) # 将光标置于文档开头（可选）

    def closeEvent(self, event):
        """关闭应用程序时的处理"""
        try:
            # 停止所有正在进行的任务
            self.stop_reading()
            
            # 保存设置
            self.save_settings()
            
            # 关闭PDF文档
            if self.pdf_doc:
                self.pdf_doc.close()
            
            # 确保音频播放器停止（异步）
            if self.audio_player:
                print("stop vodio 2")
                future = asyncio.run_coroutine_threadsafe(
                    self.audio_player.stop_current_audio(),
                    self.loop
                )
                try:
                    # 等待异步操作完成，但设置超时
                    future.result(timeout=5.0)
                except Exception as e:
                    print(f"Warning: Audio cleanup timed out or failed: {e}")

            # 清理异步相关资源
            if self.loop and self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=1.0)
            if self.executor:
                self.executor.shutdown(wait=False)
                
        except Exception as e:
            print(f"关闭时发生错误: {e}")
            
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置全局字体
    app_font = QFont()
    # 优先使用思源黑体, 然后是PingFang SC (macOS/iOS), Segoe UI (Windows), SF Pro (macOS/iOS), Microsoft YaHei (Windows fallback)
    app_font.setFamily("思源黑体 CN, Source Han Sans CN, PingFang SC, Segoe UI, SF Pro, Microsoft YaHei") 
    app_font.setPointSize(9) # 稍小的字号以实现紧凑界面
    QApplication.setFont(app_font)
    
    window = AutoReaderApp()
    window.show()
    sys.exit(app.exec_())