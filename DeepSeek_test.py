#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大模型成语接龙游戏
接入 https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
"""

import requests
import json
import random
import os
from openai import OpenAI

class DeepSeekIdiomChain:
    def __init__(self, api_key, idiom_file_path, model_name="deepseek-v3"):
        """初始化游戏"""
        self.api_key = api_key
        self.api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        self.model_name = model_name
        self.idioms = self.load_idioms(idiom_file_path)
        self.history = []
        self.used_idioms = set()
        self.player_score = 0
        self.ai_score = 0
        self.round_count = 0
        self.client = None
        self.init_openai_client()
    
    def init_openai_client(self):
        """初始化OpenAI客户端"""
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            print("[OK] OpenAI客户端初始化成功")
        except Exception as e:
            print(f"[X] OpenAI客户端初始化失败: {e}")
            self.client = None
    
    def load_idioms(self, file_path):
        """从文件加载成语数据"""
        idioms = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 解析CSV格式的行
                    parts = line.split('","')
                    if len(parts) >= 3:
                        # 提取成语（从parts[0]中提取，在引号后面的部分）
                        idiom_raw = parts[0].split('"')[-1]
                        # 提取拼音（第二个字段）
                        pinyin = parts[1].strip('"')
                        
                        idioms[idiom_raw] = {
                            'pinyin': pinyin,
                            'first_char': idiom_raw[0],  # 首字
                            'last_char': idiom_raw[-1],  # 尾字
                            'length': len(idiom_raw)
                        }
            
            print(f"成功加载 {len(idioms)} 个成语")
            return idioms
            
        except FileNotFoundError:
            print(f"错误：找不到文件 {file_path}")
            return {}
        except Exception as e:
            print(f"加载成语文件时出错：{e}")
            return {}
    
    def get_completion(self, prompt):
        """调用大模型API获取回复"""
        # 优先使用OpenAI客户端
        if self.client:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一个成语接龙游戏助手，只需要根据给定的尾字接一个合适的四字成语，不要添加任何解释。"
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=100
                )
                return completion.choices[0].message.content.strip()
            except Exception as e:
                print(f"OpenAI客户端调用失败: {e}")
                # 回退到requests方式
                return self.get_completion_requests(prompt)
        else:
            # 使用requests方式
            return self.get_completion_requests(prompt)
    
    def get_completion_requests(self, prompt):
        """使用requests调用API（备用方式）"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model_name,  # 使用指定的模型
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个成语接龙游戏助手，只需要根据给定的尾字接一个合适的四字成语，不要添加任何解释。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"API调用失败: {e}")
            return None
    
    def is_valid_idiom(self, idiom):
        """检查成语是否有效"""
        # 检查是否为四字成语
        if len(idiom) != 4:
            return False
        
        # 检查是否在成语库中
        if idiom not in self.idioms:
            return False
        
        # 检查是否已经使用过
        if idiom in self.used_idioms:
            return False
        
        return True
    
    def can_connect(self, idiom1, idiom2):
        """检查两个成语是否可以接龙（按字接龙）"""
        if idiom1 not in self.idioms or idiom2 not in self.idioms:
            return False
        
        # 获取两个成语的尾字和首字
        last_char = self.idioms[idiom1]['last_char']
        first_char = self.idioms[idiom2]['first_char']
        
        # 检查尾字和首字是否相同（严格按字接龙）
        return last_char == first_char
    
    def extract_idiom_from_response(self, response):
        """从大模型回复中提取成语"""
        response = response.strip()
        
        # 尝试直接匹配四字成语
        import re
        idiom_pattern = r'[\u4e00-\u9fa5]{4}'
        idioms = re.findall(idiom_pattern, response)
        
        # 检查是否有合法的成语
        for idiom in idioms:
            if self.is_valid_idiom(idiom):
                return idiom
        
        # 检查回复中是否包含"放弃"
        if '放弃' in response:
            return None
        
        return None
    
    def start_game(self):
        """开始游戏"""
        if not self.idioms:
            print("成语库为空，无法开始游戏")
            return
        
        print("=" * 60)
        print("🎯 大模型成语接龙游戏")
        print("=" * 60)
        print("【游戏规则】")
        print("1. 每个成语必须是四字成语")
        print("2. 后一个成语的首字必须与前一个成语的尾字完全相同")
        print("3. 不能重复使用成语")
        print("4. 玩家和大模型轮流进行接龙")
        print("5. 如果成语不在成语库中或无法接龙，则判负")
        print("\n【操作说明】")
        print("- 输入成语进行接龙")
        print("- 输入'退出'结束游戏")
        print("- 输入'提示'查看可选成语")
        print("=" * 60)
        
        # 随机选择一个起始成语
        self.current_idiom = random.choice(list(self.idioms.keys()))
        self.used_idioms.add(self.current_idiom)
        self.history.append(f"系统: {self.current_idiom}")
        
        print(f"\n【第1回合】")
        print(f"起始成语：{self.current_idiom}")
        print(f"尾字：'{self.current_idiom[-1]}'")
        
        # 决定谁先手
        first_turn = random.choice(['player', 'ai'])
        if first_turn == 'player':
            print("\n玩家先手！")
            self.player_turn()
        else:
            print("\n大模型先手！")
            self.ai_turn()
    
    def player_turn(self):
        """玩家回合"""
        self.round_count += 1
        print(f"\n【第{self.round_count}回合 - 玩家】")
        print(f"当前尾字：'{self.current_idiom[-1]}'")
        
        user_input = input("请输入成语：").strip()
        
        if user_input.lower() in ['退出', 'quit', 'exit']:
            print("玩家主动结束游戏！")
            self.show_final_result()
            return
        
        if user_input == '提示':
            self.show_hint()
            self.player_turn()
            return
        
        if not user_input:
            print("请输入有效的成语")
            self.player_turn()
            return
        
        # 验证用户输入的成语
        if not self.is_valid_idiom(user_input):
            if len(user_input) != 4:
                print(f"错误：'{user_input}' 不是四字成语")
            elif user_input not in self.idioms:
                print(f"错误：'{user_input}' 不在成语库中！你输了！")
            else:
                print(f"错误：'{user_input}' 已经使用过！你输了！")
            self.show_final_result()
            return
        
        # 检查是否可以接龙
        if not self.can_connect(self.current_idiom, user_input):
            print(f"错误：'{user_input}' 不能接在 '{self.current_idiom}' 后面")
            print(f"需要以 '{self.current_idiom[-1]}' 开头的成语！你输了！")
            self.show_final_result()
            return
        
        # 成功接龙
        self.used_idioms.add(user_input)
        self.current_idiom = user_input
        self.player_score += 1
        self.history.append(f"玩家: {user_input}")
        
        print(f"[OK] 成功接龙！玩家得分：{self.player_score}")
        
        # 检查是否还有可用的成语
        available = self.get_available_options()
        if not available:
            print("\n⚠ 没有更多可用的成语了！")
            self.show_final_result()
            return
        
        # AI回合
        self.ai_turn()
    
    def ai_turn(self):
        """AI回合"""
        print(f"\n【第{self.round_count}回合 - 大模型】")
        print("大模型正在思考...")
        
        # 构建提示
        prompt = f"请接一个以'{self.current_idiom[-1]}'开头的四字成语，不要重复使用以下成语：{', '.join(self.used_idioms)}"
        
        # 调用大模型
        ai_response = self.get_completion(prompt)
        print(f"大模型回复：{ai_response}")
        
        # 从回复中提取成语
        idiom = self.extract_idiom_from_response(ai_response)
        
        if idiom is None:
            print("大模型无法接龙，放弃回合！玩家获胜！")
            self.show_final_result()
            return
        
        if not self.is_valid_idiom(idiom):
            print(f"大模型给出的成语 '{idiom}' 无效！玩家获胜！")
            self.show_final_result()
            return
        
        if not self.can_connect(self.current_idiom, idiom):
            print(f"大模型给出的成语 '{idiom}' 无法接龙！玩家获胜！")
            self.show_final_result()
            return
        
        # 成功接龙
        self.used_idioms.add(idiom)
        self.current_idiom = idiom
        self.ai_score += 1
        self.history.append(f"大模型: {idiom}")
        
        print(f"[OK] 大模型成功接龙！AI得分：{self.ai_score}")
        
        # 检查是否还有可用的成语
        available = self.get_available_options()
        if not available:
            print("\n⚠ 没有更多可用的成语了！")
            self.show_final_result()
            return
        
        # 玩家回合
        self.player_turn()
    
    def show_hint(self):
        """显示提示"""
        available = self.get_available_options()
        if available:
            print(f"提示：可用的成语有 {len(available)} 个")
            if len(available) <= 10:
                print("可选成语：" + "、".join(available))
            else:
                # 随机显示10个
                sample = random.sample(available, 10)
                print("部分可选成语：" + "、".join(sample))
        else:
            print("没有可用的提示了！")
    
    def get_available_options(self):
        """获取当前可用的接龙选项"""
        available = []
        last_char = self.current_idiom[-1]
        
        for idiom in self.idioms:
            if idiom not in self.used_idioms and idiom[0] == last_char:
                available.append(idiom)
        
        return available
    
    def show_final_result(self):
        """显示最终结果"""
        print("\n" + "=" * 60)
        print("游戏结束！")
        print("=" * 60)
        print(f"\n【最终得分】")
        print(f"玩家得分：{self.player_score}")
        print(f"AI得分：{self.ai_score}")
        
        if self.player_score > self.ai_score:
            print("\n[*] 恭喜玩家获胜！")
        elif self.player_score < self.ai_score:
            print("\n[#] 大模型获胜！")
        else:
            print("\n[=] 平局！")
        
        print(f"\n【游戏统计】")
        print(f"总回合数：{self.round_count}")
        print(f"使用成语数量：{len(self.used_idioms)}")
        print(f"使用过的成语：{'、'.join(list(self.used_idioms))}")
        print("=" * 60)

def main():
    """主函数"""
    print("欢迎使用大模型成语接龙游戏！")
    
    # 优先从环境变量获取API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    
    if api_key:
        print("[OK] 从环境变量获取API Key成功")
    else:
        print("请输入你的API Key:")
        api_key = input().strip()
    
    if not api_key:
        print("API Key不能为空")
        return
    
    # 选择模型
    print("\n请选择使用的模型：")
    print("1. deepseek-v3 (推荐)")
    print("2. deepseek-r1")
    
    model_choice = input("请输入选项编号 (1-2): ").strip()
    
    if model_choice == "1":
        model_name = "deepseek-v3"
    elif model_choice == "2":
        model_name = "deepseek-r1"
    else:
        print("无效选项，使用默认模型 deepseek-v3")
        model_name = "deepseek-v3"
    
    print(f"\n[OK] 选择模型：{model_name}")
    
    # 成语文件路径
    idiom_file = "RAG/chinese-idiom-db-master/chinese-idioms-12976.txt"
    
    game = DeepSeekIdiomChain(api_key, idiom_file, model_name)
    game.start_game()

if __name__ == "__main__":
    main()