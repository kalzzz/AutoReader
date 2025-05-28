import asyncio
import re

def get_pdf_outline(pdf_doc):
    """Extracts the outline (table of contents) from the PDF document object."""
    if not pdf_doc:
        return []
    outline_data = []
    toc = pdf_doc.get_toc(simple=True)
    for toc_level, title, page_num_1_based in toc:
        current_level = toc_level - 1
        page_to_store = page_num_1_based if page_num_1_based > 0 else None
        item_data = {'original_title': title, 'page': page_to_store, 'level': current_level}
        if page_to_store is None:
            item_data['display_title'] = f"{title} (无有效页码)"
        else:
            item_data['display_title'] = title
        outline_data.append(item_data)
    print(f"提取到 {len(outline_data)} 个目录项")
    
    if not outline_data:
        print("未找到目录，尝试通过正则匹配生成伪目录...")

        # - \s 表示匹配任意空白字符（包括空格、制表符等），
        # - \d 表示匹配任意阿拉伯数字（0-9），
        # - 一二三四五六七八九十百千零〇 表示匹配这些常见的中文数字。
        # 中括号 [] 表示匹配其中任意一个字符，后面的加号 + 表示匹配前面字符集合中一个或多个字符。
        pattern = re.compile(r"第[\s\d一二三四五六七八九十百千零〇]+[章节篇回卷][^\n\r]{0,30}", re.UNICODE)
        for page_idx in range(pdf_doc.page_count):
            try:
                page = pdf_doc.load_page(page_idx)
                text = page.get_text("text")
                for match in pattern.finditer(text):
                    title = match.group().strip()
                    outline_data.append({
                        'original_title': title,
                        'page': page_idx + 1,
                        'level': 0,
                        'display_title': title
                    })
            except Exception as e:
                print(f"第{page_idx+1}页提取伪目录失败: {e}")
        print(f"正则匹配生成 {len(outline_data)} 个伪目录项")

    return outline_data

# async def extract_text_chunks_from_range(pdf_doc_obj_gui, start_page_0_indexed, end_page_0_indexed,
#                                        chunk_length=50, stop_reading_flag=None,
#                                        text_cleaner_func=None):
#     """异步生成文本块"""
#     try:
#         current_page_idx = start_page_0_indexed
#         text_buffer = ""
#         paragraph_index = 0
#         last_page_idx_for_chunk_assignment = start_page_0_indexed

#         while current_page_idx <= end_page_0_indexed and (not stop_reading_flag or not stop_reading_flag.is_set()):
#             # 异步加载页面文本
#             try:
#                 page = pdf_doc_obj_gui.load_page(current_page_idx)
#                 text = page.get_text("text")
                
#                 # 处理当前页面的文本
#                 sentences = re.split(r'([。！？；])', text)
#                 for i in range(0, len(sentences), 2):
#                     if stop_reading_flag and stop_reading_flag.is_set():
#                         return
                        
#                     sentence = sentences[i]
#                     punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
#                     complete_sentence = sentence + punctuation
                    
#                     if complete_sentence.strip():
#                         # 清理文本
#                         cleaned_text = text_cleaner_func(complete_sentence.strip()) if text_cleaner_func else complete_sentence.strip()
                        
#                         # 创建并产出文本块
#                         chunk_data = {
#                             "text": complete_sentence.strip(),
#                             "tts_text": cleaned_text,
#                             "page": current_page_idx + 1,  # 转换为1基页码
#                             "paragraph": paragraph_index
#                         }
#                         yield chunk_data
#                         paragraph_index += 1
                        
#                         # 允许其他协程执行
#                         await asyncio.sleep(0)
                
#                 current_page_idx += 1
                
#             except Exception as e:
#                 print(f"处理页面 {current_page_idx} 时出错: {e}")
#                 current_page_idx += 1
#                 continue
            
#     except Exception as e:
#         print(f"生成文本块时出错: {e}")
#         return

async def extract_text_chunks_from_range(pdf_doc_obj_gui, start_page_0_indexed, end_page_0_indexed,
                                       chunk_length=50, stop_reading_flag=None,
                                       text_cleaner_func=None):
    """异步生成文本块"""
    try:
        current_page_idx = start_page_0_indexed
        text_buffer = ""
        paragraph_index = 0
        # last_page_idx_for_chunk_assignment = start_page_0_indexed # This variable was unused

        punctuation_pattern = re.compile(r'[。！？；]')

        # 阿拉伯数字到中文数字的映射
        arabic_to_chinese = {
            '0': '零',
            '1': '一',
            '2': '二',
            '3': '三',
            '4': '四',
            '5': '五',
            '6': '六',
            '7': '七',
            '8': '八',
            '9': '九'
        }

        while current_page_idx <= end_page_0_indexed and (not stop_reading_flag or not stop_reading_flag.is_set()):
            # 异步加载页面文本
            try:
                page = pdf_doc_obj_gui.load_page(current_page_idx)
                text = page.get_text("text")

                # 将阿拉伯数字替换为中文数字
                for arabic, chinese in arabic_to_chinese.items():
                    text = text.replace(arabic, chinese)
                text_buffer += text # Append text from current page to buffer

                # Process text_buffer to create chunks
                # Continue chunking as long as buffer is long enough AND we can find a chunk boundary
                while len(text_buffer) >= chunk_length:
                    if stop_reading_flag and stop_reading_flag.is_set():
                        return

                    # Search for the first punctuation after chunk_length - 1
                    search_start_index = chunk_length - 1
                    match = punctuation_pattern.search(text_buffer, search_start_index)

                    if match:
                        # Punctuation found, chunk ends at punctuation
                        chunk_end_index = match.end() # end() gives the index after the match
                        chunk_text = text_buffer[:chunk_end_index]
                        text_buffer = text_buffer[chunk_end_index:]

                        if chunk_text.strip():
                            # Clean and yield the chunk
                            cleaned_text = text_cleaner_func(chunk_text.strip()) if text_cleaner_func else chunk_text.strip()

                            chunk_data = {
                                "text": chunk_text.strip(),
                                "tts_text": cleaned_text,
                                "page": current_page_idx + 1,  # Assign current page number (approximation)
                                "paragraph": paragraph_index
                            }
                            yield chunk_data
                            paragraph_index += 1

                            # Allow other协程执行
                            await asyncio.sleep(0)

                    else:
                        # No punctuation found after chunk_length in the current buffer.
                        # If this is the last page, yield the remaining buffer.
                        # Otherwise, break inner loop to load next page and continue searching.
                        if current_page_idx == end_page_0_indexed:
                            # Last page, yield remaining text as the final chunk
                            chunk_text = text_buffer
                            text_buffer = ""

                            if chunk_text.strip():
                                cleaned_text = text_cleaner_func(chunk_text.strip()) if text_cleaner_func else chunk_text.strip()
                                chunk_data = {
                                    "text": chunk_text.strip(),
                                    "tts_text": cleaned_text,
                                    "page": current_page_idx + 1, # Assign last page number (approximation)
                                    "paragraph": paragraph_index
                                }
                                yield chunk_data
                                # paragraph_index += 1 # No need to increment for the final chunk

                            # Break inner loop as all text is processed or yielded
                            break
                        else:
                            # Not the last page, break inner loop to load more text from the next page
                            break # Break the inner while loop, outer loop continues

                current_page_idx += 1

            except Exception as e:
                print(f"处理页面 {current_page_idx} 时出错: {e}")
                current_page_idx += 1
                continue

        # The remaining text in text_buffer should be handled by the 'last page' logic in the inner loop.
        # No need for an extra check here unless there's a scenario where the inner loop might exit
        # with remaining text *before* hitting the last page logic (e.g., stop_reading_flag set).
        # However, the stop_reading_flag check is at the start of loops and before yielding, so it should be fine.

    except Exception as e:
        print(f"生成文本块时出错: {e}")
        # return # Removed return here to allow outer exception to be handled if needed, though it also returns