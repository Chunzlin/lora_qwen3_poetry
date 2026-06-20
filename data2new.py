import json

def data2all(of='宋-苏轼-创作背景_rwj.json', nf='宋-苏轼-创作背景2_rwj.json',bg='背景'):
    '''
     {
        "uuid": "5081ad9e4b537475856f8973b9f4e304",
        "title": "水龙吟 楚山修竹如云",
        "author": "苏轼",
        "content": "楚山修竹如云，异材秀出千林表。\n龙须半翦，凤膺微涨，玉肌匀绕。\n木落淮南，雨晴云梦，月明风袅。\n自中郎不见，桓伊去后，知孤负、秋多少。\n闻道岭南太守，后堂深、绿珠娇小。\n绮窗学弄，梁州初遍，霓裳未了。\n嚼征含宫，泛商流羽，一声云杪。\n为使君洗尽，蛮风瘴雨，作霜天晓。",
        "dynasty": "宋",
        "collection_info": null,
        "诗体": "词",
        "时间地点": "北宋绍圣元年（1094年）- 绍圣四年（1097年），惠州",
        "前序事件": "新党执政，被贬惠州",
        "情景心情": "岭南风光，虽贬犹乐",
        "关键意象": "江水, 夕阳, 古迹, 英雄, 历史",
        "诗歌主旨": "借历史遗迹抒发对历史兴衰的感慨和人生思考",
        "风格标签": "豪放, 怀古, 苍凉",
        "用典说明": "引用屈原《离骚》; 引用晋王徽之与桓伊的典故; 引用绿珠坠楼典故",
        "情感变化": "由眼前实景，转入历史回忆，抒发怀古幽情",
        "生成工具": "Trae CN",
        "整理人": "rwj"
    },
    '''
    keys = ["时间地点", "前序事件", "情景心情", "关键意象", "诗歌主旨", "风格标签", "用典说明", "情感变化"]

    with open(of, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if bg in data[0]:
        print("数据已包含背景字段，无需转换")
        return
    
    for x in data:
        x[bg] = {}
        for k in keys:
            x[bg][k] = x.pop(k)

    with open(nf, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False,indent=4)

    print(f"数据转换完成，保存为{nf}")
    return

def data2pr(nf, poetry_raw='poetry_raw.jsonl', bg='背景'):
    with open(nf, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for x in data:
        bg_str = ' '.join([f'{k}，{v}' for k,v in x.pop(bg).items()])

        messages = [
            {"role": "system", "content": "你是一个才华横溢的诗人，选择模仿苏轼，根据给定的创作背景生成一首诗词。"},
            {"role": "user", "content": bg_str},
            {"role": "assistant", "content": f"{x['title']}/n/n{x['content']}"}
        ]
        x["messages"] = messages

    with open(poetry_raw, 'w', encoding='utf-8') as f:
        for x in data:
            messages = {'messages':x["messages"] }
            json.dump(messages, f, ensure_ascii=False)
            f.write("\n")

    print("数据转换完成，保存为poetry_raw.jsonl")
    return

if __name__ == "__main__":
    of = '宋-苏轼-创作背景_rwj.json' # 如果你的数据没有“背景”字段，将文件名改成你的原始数据文件
    nf = '宋-苏轼-创作背景2_rwj.json' # 如果你的数据已经有“背景”字段，将文件名改成你的原始数据文件
    poetry_raw = 'poetry_raw.jsonl'

    # 如果你是“创作背景”字段，bg='创作背景'；如果你是“背景”字段，bg='背景'（默认）
    data2all(of, nf, bg='背景') # # 如果你的数据已经有“背景”字段，注释掉这行
    data2pr(nf, poetry_raw, bg='背景')  