import os
import json
import torch
import numpy as np

from collections import namedtuple
from model import BertNer
from seqeval.metrics.sequence_labeling import get_entities
from transformers import BertTokenizer


def get_args(args_path, args_name=None):
    with open(args_path, "r", encoding="utf-8") as fp:
        args_dict = json.load(fp)
    # 注意args不可被修改了
    args = namedtuple(args_name, args_dict.keys())(*args_dict.values())
    return args


class Predictor:
    def __init__(self, data_name):
        self.data_name = data_name
        self.ner_args = get_args(os.path.join("./checkpoint/{}/".format(data_name), "ner_args.json"), "ner_args")
        self.ner_id2label = {int(k): v for k, v in self.ner_args.id2label.items()}
        self.tokenizer = BertTokenizer.from_pretrained(self.ner_args.bert_dir)
        self.max_seq_len = self.ner_args.max_seq_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ner_model = BertNer(self.ner_args)
        self.ner_model.load_state_dict(torch.load(os.path.join(self.ner_args.output_dir, "pytorch_model_ner.bin"), map_location="cpu"))
        self.ner_model.to(self.device)
        self.data_name = data_name

    def ner_tokenizer(self, text):
        # print("文本长度需要小于：{}".format(self.max_seq_len))
        tokenized_input = self.tokenizer(text)
        number_of_tokens = len(tokenized_input['input_ids'])
        crop = 0
        if number_of_tokens > (self.max_seq_len - 2):
            crop = int((number_of_tokens - self.max_seq_len - 2) / 2)
        # text = text[:self.max_seq_len - 2]
        # text = ["[CLS]"] + [i for i in text] + ["[SEP]"]
        # tmp_input_ids = self.tokenizer.convert_tokens_to_ids(text)
        # input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
        input_ids = tokenized_input['input_ids'][crop:number_of_tokens - crop]
        # attention_mask = [1] * len(tmp_input_ids) + [0] * (self.max_seq_len - len(tmp_input_ids))
        attention_mask = tokenized_input['attention_mask'][crop:number_of_tokens - crop]
        input_ids = torch.tensor(np.array([input_ids]))
        attention_mask = torch.tensor(np.array([attention_mask]))
        return input_ids, attention_mask

    def ner_predict(self, text):
        input_ids, attention_mask = self.ner_tokenizer(text)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        output = self.ner_model(input_ids, attention_mask)
        attention_mask = attention_mask.detach().cpu().numpy()
        length = sum(attention_mask[0])
        logits = output.logits
        logits = logits[0][1:length - 1]
        logits = [self.ner_id2label[i] for i in logits]
        print(logits)
        entities = get_entities(logits)
        result = {}
        for ent in entities:
            ent_name = ent[0]
            ent_start = ent[1]
            ent_end = ent[2]
            if ent_name not in result:
                result[ent_name] = [("".join(text[ent_start:ent_end + 1]), ent_start, ent_end)]
            else:
                result[ent_name].append(("".join(text[ent_start:ent_end + 1]), ent_start, ent_end))
        return result


if __name__ == "__main__":
    data_name = "fyp"
    predictor = Predictor(data_name)
    if data_name == "dgre":
        texts = [
            "492号汽车故障报告故障现象一辆车用户用水清洗发动机后，在正常行驶时突然产生铛铛异响，自行熄火",
            "故障现象：空调制冷效果差。",
            "原因分析：1、遥控器失效或数据丢失;2、ISU模块功能失效或工作不良;3、系统信号有干扰导致。处理方法、体会：1、检查该车发现，两把遥控器都不能工作，两把遥控器同时出现故障的可能几乎是不存在的，由此可以排除遥控器本身的故障。2、检查ISU的功能，受其控制的部分全部工作正常，排除了ISU系统出现故障的可能。3、怀疑是遥控器数据丢失，用诊断仪对系统进行重新匹配，发现遥控器匹配不能正常进行。此时拔掉ISU模块上的电源插头，使系统强制恢复出厂设置，再插上插头，发现系统恢复，可以进行遥控操作。但当车辆发动在熄火后，遥控又再次失效。4、查看线路图发现，在点火开关处安装有一钥匙行程开关，当钥匙插入在点火开关内，处于ON位时，该开关接通，向ISU发送一个信号，此时遥控器不能进行控制工作。当钥匙处于OFF位时，开关断开，遥控器恢复工作，可以对门锁进行控制。如果此开关出现故障，也会导致遥控器不能正常工作。同时该行程开关也控制天窗的自动回位功能。测试天窗发现不能自动回位。确认该开关出现故障",
            "原因分析：1、发动机点火系统不良;2、发动机系统油压不足;3、喷嘴故障;4、发动机缸压不足;5、水温传感器故障。",
        ]
    elif data_name == "duie":
        texts = [
            "歌曲《墨写你的美》是由歌手冷漠演唱的一首歌曲",
            "982年，阎维文回到山西，隆重地迎娶了刘卫星",
            "王皃姁为还是太子的刘启生了二个儿子，刘越（汉景帝第11子）、刘寄（汉景帝第12子）",
            "数据分析方法五种》是2011年格致出版社出版的图书，作者是尤恩·苏尔李",
            "视剧《不可磨灭》是导演潘培成执导，刘蓓、丁志诚、李洪涛、丁海峰、雷娟、刘赫男等联袂主演",
        ]
    elif data_name == "fyp":
        texts = [
            "Job Description: At Bank of America, we are guided by a common purpose to help make financial lives better through the power of every connection. Responsible Growth is how we run our company and how we deliver for our clients, teammates, communities and shareholders every day. One of the keys to driving Responsible Growth is being a great place to work for our teammates around the world. We’re devoted to being a diverse and inclusive workplace for everyone. We hire individuals with a broad range of backgrounds and experiences and invest heavily in our teammates and their families by offering competitive benefits to support their physical, emotional, and financial well-being. Bank of America believes both in the importance of working together and offering flexibility to our employees. We use a multi-faceted approach for flexibility, depending on the various roles in our organization. Working at Bank of America will give you a great career with opportunities to learn, grow and make an impact, along with the power to make a difference. Join us! Job Description: This job is responsible for developing and delivering complex requirements to accomplish business goals. Key responsibilities of the job include ensuring that software is developed to meet functional/non-functional requirements, coding solutions, and unit testing. Job expectations include an awareness of the development practices in the industry. The SME will leverage Mortgages securities risk subject matter expertise and Python/Quartz skills to onboard the Repo Mortgage securities in the strategic SFRC end of day framework. Responsibilities: - Codes solutions and unit test to deliver a requirement/story per the defined acceptance criteria - Executes automated test suites (integration, regression, performance); collect results and flag issues - Documents and communicates required information for deployment, maintenance, support, and business functionality - Adheres to team delivery/release process and cadence pertaining to code deployment and release - Contributes to story refinement and definition of requirements Required Skills: Senior Resource with Good Experience with Java Desired Skills: SQL Skills: - Application Development - Automation - Collaboration - DevOps Practices - Solution Design - Agile Practices - Architecture - Result Orientation - Solution Delivery Process - User Experience Design - Analytical Thinking - Data Management - Risk Management - Technical Strategy Development - Test Engineering Minimum Education Requirement: Bachelors degree or equivalent work experience Shift: 1st shift (United States of America) Hours Per Week: 40 ",
            "Join our team, located in the stunning city of Vancouver, Canada, where we foster a culture of growth and diversity. Our team is responsible for the highly scalable Microsoft Forms service, which serves customers worldwide with SAAS architecture. With Microsoft Forms, information workers and educators can easily collect survey and assessment results, and generate insights through automatic data analysis and AI, whether on desktop or mobile.  We Are Looking To Hire a Software Engineer I Who Is Seeking New Challenges And Opportunities For Growth In One Or More Of The Following Areas - Scalable SAAS service utilizing public cloud services. - Scalable web client utilizing open technology like React, Redux, Webpack. - Big data and AI. - Technical leadership and collaboration with Microsoft teams in the US and worldwide. - Improving large-scale system architecture and infrastructure. - Customer obsession and engagement. If this is you, we invite you to join our mission and be part of our exciting journey. Check out our latest blog post and videos to see what Microsoft Forms team in Vancouver has been up to lately: Microsoft Forms: Connecting with Customers to Empower Innovation - Microsoft Vancouver Microsoft’s mission is to empower every person and every organization on the planet to achieve more. As employees we come together with a growth mindset, innovate to empower others, and collaborate to realize our shared goals. Each day we build on our values of respect, integrity, and accountability to create a culture of inclusion where everyone can thrive at work and beyond. Embody our Culture and Values Responsibilities - Understand user requirements and design solutions to meet those needs. - Write and debug code for various software products or features. - Deploy features and monitor their performance in live services. - Follow engineering best practices throughout the software development lifecycle. - Act as a Designated Responsible Individual (DRI) for simple system/product/service issues, escalating complex problems as needed. - Reviews current developments and proactively seeks new knowledge that will improve the availability, reliability, efficiency, observability, and performance of products while also driving consistency in monitoring and operations at scale. Qualifications Required Qualifications - Bachelor's Degree in Computer Science, or related technical discipline with proven experience coding in languages including, but not limited to, C, C++, C#, Java, JavaScript, or Python - - OR equivalent experience. Preferred Qualifications - -  Bachelor's Degree in Computer Science or related technical field AND 1+ year(s) technical engineering experience with coding in languages including, but not limited to, C, C++, C#, Java, JavaScript, or Python - OR Master's Degree in Computer Science or related technical field with proven experience coding in languages including, but not limited to, C, C++, C#, Java, JavaScript, or Python - OR equivalent experience. Software Engineering IC2 - The typical base pay range for this role across Canada is CAD $61,500 - CAD $121,200 per year. Find Additional Pay Information Here https://careers.microsoft.com/v2/global/en/canada-pay-information.html Microsoft will accept applications for the role until May 14, 2023. Microsoft is an equal opportunity employer. Consistent with applicable law, all qualified applicants will receive consideration for employment without regard to age, ancestry, citizenship, color, family or medical care leave, gender identity or expression, genetic information, immigration status, marital status, medical condition, national origin, physical or mental disability, political affiliation, protected veteran or military status, race, ethnicity, religion, sex (including pregnancy), sexual orientation, or any other characteristic protected by applicable local laws, regulations and ordinances. If you need assistance and/or a reasonable accommodation due to a disability during the application process, read more about requesting accommodations. "
        ]
    for text in texts:
        ner_result = predictor.ner_predict(text)
        print("文本>>>>>：", text)
        print("实体>>>>>：", ner_result)
        print("="*100)


