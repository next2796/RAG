#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成语库双模型测试工具
实现混合检索策略：语义检索（70%）+ 关键词检索（30%）
设计3类测试问题：事实型/推理型/多跳推理型
输出两个模型的对比结果和图表
"""

import os
import json
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class IdiomTestTool:
    def __init__(self, idiom_file_path):
        """初始化测试工具"""
        self.idioms = self.load_idioms(idiom_file_path)
        self.idiom_list = list(self.idioms.keys())
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.init_tfidf()
    
    def load_idioms(self, file_path):
        """加载三国演义文本数据"""
        idioms = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 简单处理，提取四字词语
                # 使用正则表达式匹配四字词语
                import re
                # 匹配连续的四个汉字
                four_char_words = re.findall(r'[\u4e00-\u9fa5]{4}', content)
                
                # 去重并构建词语库
                for word in set(four_char_words):
                    idioms[word] = {
                        'pinyin': '',
                        'first_char': word[0],
                        'last_char': word[-1],
                        'length': len(word)
                    }
            print(f"成功加载 {len(idioms)} 个四字词语")
            return idioms
        except Exception as e:
            print(f"加载文本文件时出错：{e}")
            return {}
    
    def init_tfidf(self):
        """初始化TF-IDF向量器"""
        try:
            texts = [idiom for idiom in self.idiom_list]
            self.tfidf_vectorizer = TfidfVectorizer(analyzer='char')
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            print("TF-IDF向量器初始化成功")
        except Exception as e:
            print(f"初始化TF-IDF向量器失败：{e}")
    
    def keyword_search(self, query, top_k=10):
        """关键词检索"""
        if not self.tfidf_vectorizer:
            return []
        
        query_vec = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        sorted_indices = np.argsort(similarities)[::-1]
        results = []
        for i in sorted_indices[:top_k]:
            idiom = self.idiom_list[i]
            results.append((idiom, similarities[i]))
        
        return results
    
    def semantic_search(self, query, top_k=10):
        """语义检索（简化版，实际应用中可使用embedding模型）"""
        results = []
        for idiom in self.idiom_list:
            # 简单的语义相似度计算
            score = 0
            for char in query:
                if char in idiom:
                    score += 1
            if score > 0:
                results.append((idiom, score / len(query)))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def hybrid_search(self, query, top_k=10):
        """混合检索：语义检索（70%）+ 关键词检索（30%）"""
        semantic_results = self.semantic_search(query, top_k * 2)
        keyword_results = self.keyword_search(query, top_k * 2)
        
        # 构建分数字典
        scores = {}
        for idiom, score in semantic_results:
            scores[idiom] = scores.get(idiom, 0) + score * 0.7
        for idiom, score in keyword_results:
            scores[idiom] = scores.get(idiom, 0) + score * 0.3
        
        # 排序并返回
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

class ModelTester:
    def __init__(self, api_key):
        """初始化模型测试器"""
        self.api_key = api_key
        self.models = {
            "deepseek-v3": self.init_model("deepseek-v3"),
            "deepseek-r1": self.init_model("deepseek-r1")
        }
    
    def init_model(self, model_name):
        """初始化模型"""
        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            return client
        except Exception as e:
            print(f"初始化模型 {model_name} 失败：{e}")
            return None
    
    def test_model(self, model_name, prompt):
        """测试模型"""
        client = self.models.get(model_name)
        if not client:
            return None, 0
        
        start_time = time.time()
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个成语专家，回答要准确简洁。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=200
            )
            response_time = time.time() - start_time
            return completion.choices[0].message.content.strip(), response_time
        except Exception as e:
            print(f"测试模型 {model_name} 失败：{e}")
            return None, time.time() - start_time

class TestCaseGenerator:
    def __init__(self, idioms):
        """初始化测试用例生成器"""
        self.idioms = idioms
        self.idiom_list = list(idioms.keys())
    
    def generate_fact_questions(self, count=10):
        """生成事实型问题"""
        questions = []
        for _ in range(count):
            idiom = random.choice(self.idiom_list)
            question = f"{idiom} 的意思是什么？"
            questions.append({
                "type": "事实型",
                "question": question,
                "target": idiom
            })
        return questions
    
    def generate_reasoning_questions(self, count=10):
        """生成推理型问题"""
        questions = []
        for _ in range(count):
            idiom = random.choice(self.idiom_list)
            question = f"用 {idiom} 造一个句子"
            questions.append({
                "type": "推理型",
                "question": question,
                "target": idiom
            })
        return questions
    
    def generate_multi_hop_questions(self, count=10):
        """生成多跳推理型问题"""
        questions = []
        for _ in range(count):
            # 找到可以接龙的成语
            idiom1 = random.choice(self.idiom_list)
            candidates = [idiom for idiom in self.idiom_list if idiom[0] == idiom1[-1] and idiom != idiom1]
            if not candidates:
                continue
            idiom2 = random.choice(candidates)
            question = f"以 {idiom1} 开头，接一个成语，然后用接的成语再造一个句子"
            questions.append({
                "type": "多跳推理型",
                "question": question,
                "target": f"{idiom1} → {idiom2}"
            })
        return questions

class ResultAnalyzer:
    def __init__(self):
        """初始化结果分析器"""
        self.results = {}
    
    def add_result(self, model_name, question_type, response, response_time, correct):
        """添加测试结果"""
        if model_name not in self.results:
            self.results[model_name] = {}
        if question_type not in self.results[model_name]:
            self.results[model_name][question_type] = {
                "responses": [],
                "response_times": [],
                "correctness": []
            }
        self.results[model_name][question_type]["responses"].append(response)
        self.results[model_name][question_type]["response_times"].append(response_time)
        self.results[model_name][question_type]["correctness"].append(1 if correct else 0)
    
    def calculate_metrics(self):
        """计算各项指标"""
        metrics = {}
        for model_name, model_results in self.results.items():
            metrics[model_name] = {}
            for question_type, results in model_results.items():
                correctness = results["correctness"]
                response_times = results["response_times"]
                metrics[model_name][question_type] = {
                    "accuracy": sum(correctness) / len(correctness) if correctness else 0,
                    "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
                    "total_questions": len(correctness)
                }
        return metrics
    
    def generate_charts(self, metrics):
        """生成图表"""
        # 准确率对比图
        self._plot_accuracy_comparison(metrics)
        # 响应时间对比图
        self._plot_response_time_comparison(metrics)
        # 综合性能对比图
        self._plot_combined_performance(metrics)
    
    def _plot_accuracy_comparison(self, metrics):
        """绘制准确率对比图"""
        plt.figure(figsize=(10, 6))
        
        question_types = list(metrics["deepseek-v3"].keys())
        models = list(metrics.keys())
        
        for i, model_name in enumerate(models):
            accuracies = [metrics[model_name][qtype]["accuracy"] for qtype in question_types]
            plt.bar(np.arange(len(question_types)) + i * 0.3, accuracies, width=0.3, label=model_name)
        
        plt.xlabel('问题类型')
        plt.ylabel('准确率')
        plt.title('不同模型的准确率对比')
        plt.xticks(np.arange(len(question_types)) + 0.3, question_types)
        plt.legend()
        plt.tight_layout()
        plt.savefig('accuracy_comparison.png')
        print("准确率对比图已保存为 accuracy_comparison.png")
    
    def _plot_response_time_comparison(self, metrics):
        """绘制响应时间对比图"""
        plt.figure(figsize=(10, 6))
        
        question_types = list(metrics["deepseek-v3"].keys())
        models = list(metrics.keys())
        
        for i, model_name in enumerate(models):
            response_times = [metrics[model_name][qtype]["avg_response_time"] for qtype in question_types]
            plt.bar(np.arange(len(question_types)) + i * 0.3, response_times, width=0.3, label=model_name)
        
        plt.xlabel('问题类型')
        plt.ylabel('平均响应时间（秒）')
        plt.title('不同模型的响应时间对比')
        plt.xticks(np.arange(len(question_types)) + 0.3, question_types)
        plt.legend()
        plt.tight_layout()
        plt.savefig('response_time_comparison.png')
        print("响应时间对比图已保存为 response_time_comparison.png")
    
    def _plot_combined_performance(self, metrics):
        """绘制综合性能对比图"""
        plt.figure(figsize=(12, 6))
        
        models = list(metrics.keys())
        question_types = list(metrics["deepseek-v3"].keys())
        
        for i, qtype in enumerate(question_types):
            plt.subplot(1, len(question_types), i + 1)
            for model_name in models:
                accuracy = metrics[model_name][qtype]["accuracy"]
                response_time = metrics[model_name][qtype]["avg_response_time"]
                plt.scatter(response_time, accuracy, label=model_name, s=100)
            plt.xlabel('响应时间（秒）')
            plt.ylabel('准确率')
            plt.title(f'{qtype} 性能')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('combined_performance.png')
        print("综合性能对比图已保存为 combined_performance.png")

    def save_results(self, metrics):
        """保存测试结果"""
        with open('test_results.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print("测试结果已保存为 test_results.json")

        # 生成文本报告
        with open('test_report.txt', 'w', encoding='utf-8') as f:
            f.write("# 三国演义四字词语模型测试报告\n\n")
            f.write("## 测试概览\n\n")
            for model_name, model_metrics in metrics.items():
                f.write(f"### {model_name}\n\n")
                for qtype, metrics_data in model_metrics.items():
                    f.write(f"- {qtype}:\n")
                    f.write(f"  - 准确率: {metrics_data['accuracy']:.4f}\n")
                    f.write(f"  - 平均响应时间: {metrics_data['avg_response_time']:.4f}秒\n")
                    f.write(f"  - 测试数量: {metrics_data['total_questions']}\n")
                f.write("\n")
        print("测试报告已保存为 test_report.txt")

def main():
    """主函数"""
    print("成语库双模型测试工具")
    print("=" * 60)
    
    # 获取API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        api_key = input("请输入你的API Key: ").strip()
        if not api_key:
            print("API Key不能为空")
            return
    
    # 加载三国演义文本
    idiom_file = "RAG/sanguoyanyi.txt"
    test_tool = IdiomTestTool(idiom_file)
    
    if not test_tool.idioms:
        print("无法加载三国演义文本")
        return
    
    # 初始化模型测试器
    tester = ModelTester(api_key)
    
    # 生成测试用例
    generator = TestCaseGenerator(test_tool.idioms)
    
    fact_questions = generator.generate_fact_questions(5)
    reasoning_questions = generator.generate_reasoning_questions(5)
    multi_hop_questions = generator.generate_multi_hop_questions(5)
    
    all_questions = fact_questions + reasoning_questions + multi_hop_questions
    print(f"生成测试用例：{len(all_questions)} 个")
    
    # 初始化结果分析器
    analyzer = ResultAnalyzer()
    
    # 测试模型
    for question in all_questions:
        print(f"\n测试问题：{question['question']}")
        print(f"问题类型：{question['type']}")
        
        # 测试两个模型
        for model_name in ["deepseek-v3", "deepseek-r1"]:
            response, response_time = tester.test_model(model_name, question['question'])
            
            if response:
                print(f"{model_name} 响应：{response}")
                print(f"响应时间：{response_time:.2f}秒")
                
                # 简单的正确性判断（实际应用中需要更复杂的评估）
                correct = question['target'] in response
                analyzer.add_result(model_name, question['type'], response, response_time, correct)
            else:
                print(f"{model_name} 测试失败")
                analyzer.add_result(model_name, question['type'], None, response_time, False)
    
    # 计算指标
    metrics = analyzer.calculate_metrics()
    
    # 生成图表
    analyzer.generate_charts(metrics)
    
    # 保存结果
    analyzer.save_results(metrics)
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("生成的文件：")
    print("- accuracy_comparison.png：准确率对比图")
    print("- response_time_comparison.png：响应时间对比图")
    print("- combined_performance.png：综合性能对比图")
    print("- test_results.json：测试结果数据")
    print("- test_report.txt：测试报告")

if __name__ == "__main__":
    main()