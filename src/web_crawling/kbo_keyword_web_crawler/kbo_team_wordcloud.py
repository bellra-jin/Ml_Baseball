import pandas as pd
from kiwipiepy import Kiwi
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# =====================================================================
# 1단계: 형태소 분석기 준비
# =====================================================================
kiwi = Kiwi()
print('라이브러리 로드 완료!')

# =====================================================================
# 2단계: CSV 데이터 불러오기
# =====================================================================
df = pd.read_csv('프로야구_제목.csv', encoding='utf-8-sig')

print(f'총 뉴스 제목 수: {len(df)}개')
print(df.head())

# =====================================================================
# 3단계: Kiwi 형태소 분석 — 명사만 추출
# =====================================================================
NOUN_TAGS = {'NN', 'NNP'}  # 일반명사 + 고유명사

nouns = []
for title in df['제목'].dropna():
    tokens = kiwi.tokenize(title)
    for token in tokens:
        if token.tag in NOUN_TAGS and len(token.form) >= 2:
            nouns.append(token.form)

print(f'추출된 명사 토큰 수: {len(nouns)}개')
print('샘플 (앞 10개):', nouns[:10])

# =====================================================================
# 4단계: 일반 불용어만 제거 — 팀명은 유지
# =====================================================================
STOPWORDS = {
    '뉴스', '기자', '관련', '발표', '내용', '경우', '문제',
    '제목', '이후', '이번', '오늘', '지난', '최근', '통해',
    '대해', '위해', '때문', '대한', '이상', '이하'
}

filtered = [w for w in nouns if w not in STOPWORDS]

print(f'불용어 제거 전: {len(nouns)}개')
print(f'불용어 제거 후: {len(filtered)}개')

# =====================================================================
# 5단계: 단어 빈도 계산
# =====================================================================
word_freq = Counter(filtered)

print('상위 20개 단어 (팀명 포함):')
for word, cnt in word_freq.most_common(20):
    bar = '|' * cnt
    print(f'  {word:10s} {bar} ({cnt}회)')

# =====================================================================
# 6단계: 워드 클라우드 시각화
# =====================================================================
FONT_PATH = r'C:\Windows\Fonts\malgun.ttf'
font_prop = fm.FontProperties(fname=FONT_PATH)
plt.rcParams['font.family'] = font_prop.get_name()

wc = WordCloud(
    font_path=FONT_PATH,
    width=900,
    height=600,
    background_color='white',
    max_words=80,
    max_font_size=120,
    colormap='tab10',
    prefer_horizontal=0.8,
)

wc.generate_from_frequencies(word_freq)

fig, ax = plt.subplots(figsize=(14, 9))
ax.imshow(wc, interpolation='bilinear')
ax.axis('off')
ax.set_title(
    '프로야구 뉴스 제목 워드 클라우드 (팀명 포함)',
    fontsize=18, pad=16, fontproperties=font_prop
)
plt.tight_layout()
plt.show()
