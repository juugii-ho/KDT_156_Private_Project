replacements = {
    'HIGH4 (하이포), 아이유': '하이포',
    'Nerd Connection' : '너드 커넥션',
    'ZICO' : '지코',
}

import pandas as pd
import re
import requests
import urllib.request
from bs4 import BeautifulSoup
import csv
import datetime
import time
from bs4 import CData
import numpy as np

datetime.datetime.today()
now = datetime.datetime.now()
nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')

def main():

    melonDF = pd.read_csv('c_melon.csv', delimiter=',', encoding='utf-8', skiprows=1, header=None)
    bugsDF = pd.read_csv('c_bugs.csv', delimiter=',', encoding='utf-8', skiprows=1, header=None)
    floDF = pd.read_csv('c_flo.csv', delimiter=',', encoding='utf-8', skiprows=1, header=None)
    vibeDF = pd.read_csv('c_vibe.csv', delimiter=',', encoding='utf-8', skiprows=1, header=None)
    genieDF = pd.read_csv('c_genie.csv', delimiter=',', encoding='utf-8', skiprows=1, header=None)
    appleDF = pd.read_csv('c_apple.csv', delimiter=',', encoding='utf-8', skiprows=1, header=None, on_bad_lines='skip')
    appleDF2 = pd.read_csv('c_apple.csv', delimiter=',', encoding='utf-8', skiprows=5, header=None, on_bad_lines='skip')

    # melonDF.shape, bugsDF.shape, floDF.shape, vibeDF.shape, genieDF.shape, appleDF.shape

    columns_to_keep = [i for i in range(0, 400, 4)]

    # iloc을 사용하여 선택한 열만 포함하는 DataFrame 생성
    appleDF3 = appleDF2.iloc[:, columns_to_keep]
    appleDF3.columns = range(100)

    appleDF4 = pd.concat([appleDF, appleDF3], axis=0).reset_index(drop=True)

    melonDF = melonDF.tail(5).T
    bugsDF = bugsDF.tail(3).T
    floDF = floDF.tail(3).T
    vibeDF = vibeDF.tail(3).T
    genieDF = genieDF.tail(3).T
    appleDF = appleDF4.tail(5).T

    melonDF.columns = ['song', 'artist', 'songNum', 'artistNum', 'albumcover']
    bugsDF.columns = ['song', 'artist', 'albumcover']
    floDF.columns = ['song', 'artist', 'albumcover']
    vibeDF.columns = ['song', 'artist', 'albumcover']
    genieDF.columns = ['song', 'artist', 'albumcover']
    appleDF.columns = ['song', 'artist', 'songENG', 'artistENG', 'albumcover']

    melonDF['artist'] = melonDF['artist'].replace(replacements)
    bugsDF['artist'] = bugsDF['artist'].replace(replacements)
    vibeDF['artist'] = vibeDF['artist'].replace(replacements)
    floDF['artist'] = floDF['artist'].replace(replacements)
    genieDF['artist'] = genieDF['artist'].replace(replacements)
    appleDF['artist'] = appleDF['artist'].replace(replacements)

    def duplicatedSum(df):
        df.song.duplicated().sum()

    def sub_df(df):
        df['artist'] = df['artist'].str.split(' &').str[0]
        df['artist'] = df['artist'].str.split(', ').str[0]

        df['song_'] = df['song'].str.replace(r'\(.*?\)', '', regex=True).str.strip()
        df['artist_'] = df['artist'].str.replace(r'\(.*?\)', '', regex=True).str.strip()

        df['songSub'] = df['song'].str.extract(r'\((.*?)\)')
        df['artistSub'] = df['artist'].str.extract(r'\((.*?)\)')

        # 한국어와 영어 분리 로직 추가
        def split_kr_en_num_jp(artist):
            kr = ''.join(re.findall('[가-힣]+', artist))
            en = ''.join(re.findall('[a-zA-Z]+', artist))
            num = ''.join(re.findall('[0-9]+', artist))
            jp = ''.join(re.findall('[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u20000-\u2A6DF]+', artist))
            return kr, en, num, jp

        # 분리된 한국어와 영어를 각 열에 할당
        for index, row in df.iterrows():
            kr, en, num, jp = split_kr_en_num_jp(row['song_'])
            if kr and en and num:
                df.at[index, 'song_KR'] = kr + num
                df.at[index, 'song_ENG'] = en + num
            elif kr and num:
                df.at[index, 'song_KR'] = kr + num
            elif en and num:
                df.at[index, 'song_ENG'] = en + num
            elif kr and en:
                df.at[index, 'song_KR'] = kr
                df.at[index, 'song_ENG'] = en
            elif kr:
                df.at[index, 'song_KR'] = kr
            elif en:
                df.at[index, 'song_ENG'] = en
            elif num:
                df.at[index, 'song_KR'] = num
            elif jp:
                df.at[index, 'song_KR'] = jp

        for index, row in df.iterrows():
            kr, en, num, jp = split_kr_en_num_jp(row['artist_'])
            if kr and en and num:
                df.at[index, 'artist_KR'] = kr + num
                df.at[index, 'artist_ENG'] = en + num
            elif kr and num :
                df.at[index, 'artist_KR'] = kr + num
            elif en and num :
                df.at[index, 'artist_ENG'] = en + num
            elif kr and en :
                df.at[index, 'artist_KR'] = kr
                df.at[index, 'artist_ENG'] = en
            elif kr:
                df.at[index, 'artist_KR'] = kr
            elif en:
                df.at[index, 'artist_ENG'] = en
            elif num:
                df.at[index, 'artist_KR'] = num
            elif jp:
                df.at[index, 'artist_KR'] = jp

        df['song_FIN'] = df['song_KR'].combine_first(df['song_ENG'])
        df['artist_FIN'] = df['artist_KR'].combine_first(df['artist_ENG'])

        return df

    melonDF = sub_df(melonDF)
    bugsDF = sub_df(bugsDF)
    genieDF = sub_df(genieDF)
    vibeDF = sub_df(vibeDF)
    floDF = sub_df(floDF)
    appleDF = sub_df(appleDF)


    def normalize_song(df):
        # 문자열이 아닌 경우를 고려하여 처리
        # 일본어와 한자를 포함하는 정규 표현식으로 수정
        df['norm_song'] = df['song_FIN'].apply(
            lambda x: re.sub('[^가-힣a-zA-Z0-9\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', '', str(x)) if isinstance(x, str) else x
        )
        df['norm_song'] = df['norm_song'].str.replace(" ", "")  # 공백 제거
        df['norm_song'] = df['norm_song'].str.replace(r'[^\w\s]', '')  # 구두점 및 특수문자 제거
        df['norm_song'] = df['norm_song'].str.lower()  # 소문자로 변환
        return df


    def normalize_artist(df):
        df['norm_artist'] = df['artist_FIN'].apply(
            lambda x: re.sub('[^가-힣a-zA-Z0-9\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', '', str(x)) if isinstance(x, str) else x
        )
        df['norm_artist'] = df['norm_artist'].str.replace(" ", "")  # 공백 제거
        df['norm_artist'] = df['norm_artist'].str.replace(r'[^\w\s]', '')  # 구두점 및 특수문자 제거
        df['norm_artist'] = df['norm_artist'].str.lower()  # 소문자로 변환
        return df

    normMelonDF = normalize_song(melonDF)
    normBugsDF = normalize_song(bugsDF)
    normFloDF = normalize_song(floDF)
    normVibeDF = normalize_song(vibeDF)
    normGenieDF = normalize_song(genieDF)
    normAppleDF = normalize_song(appleDF)

    normMelonDF = normalize_artist(normMelonDF)
    normBugsDF = normalize_artist(normBugsDF)
    normFloDF = normalize_artist(normFloDF)
    normVibeDF = normalize_artist(normVibeDF)
    normGenieDF = normalize_artist(normGenieDF)
    normAppleDF = normalize_artist(normAppleDF)

    def normalize_artist_names(df, ref_df):
        # ref_df의 artistENG에서 공백 제거 및 소문자 변환
        ref_df['artistENG'] = ref_df['artistENG'].str.replace(" ", "").str.lower()

        # ref_df를 딕셔너리로 변환 (ENG -> KR)
        artist_dict = dict(zip(ref_df['artistENG'], ref_df['norm_artist']))

        # df의 'norm_artist' 열을 확인하고, artist_dict를 사용하여 정규화
        df['norm_artist'] = df['norm_artist'].apply(lambda x: artist_dict.get(x.lower().replace(" ", ""), x))
        return df

    normGenieDF = normalize_artist_names(normGenieDF, normAppleDF)
    normMelonDF = normalize_artist_names(normMelonDF, normAppleDF)
    normVibeDF = normalize_artist_names(normVibeDF, normAppleDF)
    normFloDF = normalize_artist_names(normFloDF, normAppleDF)
    normBugsDF = normalize_artist_names(normBugsDF, normAppleDF)

    def add_suffix_except_for(df, suffix, except_for):
        df = df.rename(columns={col: f"{col}_{suffix}" if col not in except_for else col for col in df.columns})
        return df

    normMelonDF = add_suffix_except_for(normMelonDF, 'Melon', ['norm_song', 'norm_artist', 'albumcover'])
    normVibeDF = add_suffix_except_for(normVibeDF, 'Vibe', ['norm_song', 'norm_artist', 'albumcover'])
    normBugsDF = add_suffix_except_for(normBugsDF, 'Bugs', ['norm_song', 'norm_artist', 'albumcover'])
    normGenieDF = add_suffix_except_for(normGenieDF, 'Genie', ['norm_song', 'norm_artist', 'albumcover'])
    normFloDF = add_suffix_except_for(normFloDF, 'Flo', ['norm_song', 'norm_artist', 'albumcover'])
    normAppleDF = add_suffix_except_for(normAppleDF, 'Apple', ['norm_song', 'norm_artist', 'albumcover'])

    def mergeDF(df1, df2):
        # 두 DataFrame을 병합하면서 각각의 albumcover 열에 서로 다른 접미사 추가
        merged_df = pd.merge(df1, df2, on=['norm_song', 'norm_artist'], how='outer', 
                            suffixes=('_df1', '_df2'))
        
        # albumcover 열을 생성하고, df1의 albumcover가 없을 경우 df2의 albumcover로 채움
        # 접미사에 따라 열 이름 지정 필요
        if 'albumcover_df1' in merged_df and 'albumcover_df2' in merged_df:
            merged_df['albumcover'] = merged_df['albumcover_df1'].fillna(merged_df['albumcover_df2'])
            # 사용한 임시 열들을 삭제
            merged_df.drop(columns=['albumcover_df1', 'albumcover_df2'], inplace=True)
        elif 'albumcover_df1' in merged_df:
            merged_df['albumcover'] = merged_df['albumcover_df1']
            merged_df.drop(columns=['albumcover_df1'], inplace=True)
        elif 'albumcover_df2' in merged_df:
            merged_df['albumcover'] = merged_df['albumcover_df2']
            merged_df.drop(columns=['albumcover_df2'], inplace=True)

        return merged_df

    merge1  = mergeDF(normAppleDF,normMelonDF)
    merge2  = mergeDF(merge1,normBugsDF)
    merge3  = mergeDF(merge2,normGenieDF)
    merge4  = mergeDF(merge3,normFloDF)
    merge5  = mergeDF(merge4,normVibeDF)

    errorSong = merge5[merge5[['norm_song', 'norm_artist']].norm_song.duplicated(keep=False)].sort_values('norm_song')[['norm_song', 'norm_artist']]
    print('errorSong')
    print(errorSong)

    df =merge5[['norm_artist','norm_song','song_Apple', 'artist_Apple', 'song_Melon', 'artist_Melon', 'song_Bugs', 'artist_Bugs', 'song_Flo', 'artist_Flo', 'song_Genie', 'artist_Genie', 'song_Vibe', 'artist_Vibe', 'albumcover']]

    def get_first_valid(df, col_like):
        return df.filter(like=col_like).apply(lambda row: row.dropna().head(1).values[0] if not row.dropna().empty else np.nan, axis=1)

    df_copy = df.copy()

    # 복사된 데이터프레임에 대해 값을 설정합니다.
    df_copy['song'] = get_first_valid(df_copy, 'song')
    df_copy['artist'] = get_first_valid(df_copy, 'artist')

    # 'song'과 'artist'를 합쳐 'song_artist' 열 생성
    df_copy['norm_song_artist'] = df_copy.apply(lambda x: f"{x['norm_song']} {x['norm_artist']}", axis=1)

    # 결과 출력
    seleniumApple = df_copy[['song', 'artist', 'norm_song', 'norm_artist', 'norm_song_artist', 'albumcover']]
    # seleniumApple.to_csv('seleniumApple.csv', index=False, encoding='utf-8-sig')
    # seleniumApple = pd.read_csv('seleniumApple.csv', delimiter=',', encoding='utf-8', skiprows=1, header=None)

    href_values = []

    for i in range(seleniumApple.shape[0]):
        try:
            # 검색어를 사용해 URL 생성
            search_url = 'https://music.apple.com/kr/search?term=' + seleniumApple.iloc[i, 4]
            # HTTP 요청 실행
            response = requests.get(search_url)
            # 응답 내용을 기반으로 BeautifulSoup 객체 생성
            soup = BeautifulSoup(response.content.decode('utf-8-sig', 'replace'), 'html.parser')
            # 원하는 div 찾기
            div = soup.find('div', {'class': 'track-lockup__clamp-wrapper svelte-ruivs4'})

            if div is not None:
                a_tag = div.find('a', class_='click-action')
                # 'href' 속성 값 추출
                # print(a_tag['href'].split('=')[1])
                if a_tag is not None and 'href' in a_tag.attrs:

                    href_values.append(a_tag['href'].split('=')[1])
                    # print(a_tag['href'])
                else:
                    # a_tag가 없거나 'href' 속성이 없을 경우 빈 값 추가
                    href_values.append('')
                    # print(a_tag['href'])
            else:
                # div가 없을 경우 빈 값 추가
                href_values.append('')
        except Exception as e:
            #         # 예외가 발생할 경우 빈 값 추가
            href_values.append('')
            print(f"에러 발생 : {e}")
    #
    # # 결과 출력
    # href_values


    count = 1
    patternArtist = r'(?<= - )(.*?)(?= - )'
    patternSong = r'^.*?(?=\s-\s)'

    pattern_artistUK = r'(?<=– Song by )[^–]+'
    pattern_songUK = r'(?<=‎)[^–]+'

    artistAppleListKR = []
    songAppleListKR = []
    albumCover = []

    for a in href_values:
        applesongdataKR = requests.get('https://music.apple.com/kr/song/' + a)
        applesongsoupKR = BeautifulSoup(applesongdataKR.content.decode('utf-8-sig', 'replace'), 'html.parser')


        for b in applesongsoupKR:
            try:
                a = applesongsoupKR.find('picture')
                jpeg_source = a.find('source', {'type': 'image/jpeg'})

                # 해당 `<source>` 태그의 `srcset` 속성을 가져옵니다.
                srcset = jpeg_source['srcset']

                # 원하는 해상도의 URL을 찾습니다. 여기서는 "600w" 해상도입니다.
                desired_resolution = "600w"
                selected_url = None

                # srcset에서 각 URL을 파싱합니다.
                for image_info in srcset.split(','):
                    url, resolution = image_info.strip().split(' ')
                    if resolution == desired_resolution:
                        selected_url = url
                        break
                albumCover.append(selected_url)

                c = b.find('title').text
                matchAritst = re.search(patternArtist, c)
                matchSong = re.search(patternSong, c)
                if matchAritst:
                    artist = matchAritst.group()
                    artist = artist[:-4]
                    artistAppleListKR.append(artist)
                if matchSong:
                    song = matchSong.group()
                    songAppleListKR.append(song[1:])
                count +=1

            except AttributeError:
                artistAppleListKR.append('')
                songAppleListKR.append('')


    artistAppleListUK = []
    songAppleListUK = []

    for a in href_values:
        applesongdataUK = requests.get('https://music.apple.com/kr/song/' + a + '?l=en-GB')
        applesongsoupUK = BeautifulSoup(applesongdataUK.content.decode('utf-8-sig', 'replace'), 'html.parser')

        for b in applesongsoupUK:
            try:
                c = b.find('title').text
                matchAritst = re.search(pattern_artistUK, c)
                matchSong = re.search(pattern_songUK, c)
                if matchAritst:
                    artist = matchAritst.group()
                    artistAppleListUK.append(artist[:-1])
                if matchSong:
                    song = matchSong.group()
                    songAppleListUK.append(song[:-1])
                count += 1

            except AttributeError:
                artistAppleListUK.append('')
                songAppleListUK.append('')

    empty_indices = [index for index, song in enumerate(songAppleListKR[2::4]) if song == '']

    # albumCover 리스트를 조정하여 빈 행 삽입
    adjusted_album_cover = albumCover[::4]  # 기존 4의 배수로 슬라이싱

    # empty_indices에 맞게 adjusted_album_cover에 빈 행 삽입
    offset = 0  # 빈 행이 삽입된 횟수에 따른 오프셋
    for index in empty_indices:
        adjusted_index = index  # 삽입할 실제 위치
        adjusted_album_cover.insert(adjusted_index, '')  # 빈 문자열 삽입
        offset +=1  # 빈 행을 삽입할 때마다 오프셋 증가

    with open('total_KR_ENG.csv', 'a', encoding='utf-8-sig', newline='') as f:
        applewriter = csv.writer(f)
        applewriter.writerow([nowDatetime])
        applewriter.writerows([adjusted_album_cover])
        applewriter.writerows([href_values])
        applewriter.writerows([songAppleListKR[2::4]])
        applewriter.writerows([artistAppleListKR[2::4]])
        applewriter.writerows([songAppleListUK[2::4]])
        applewriter.writerows([artistAppleListUK[2::4]])

        melon = melonDF[['norm_song', 'norm_artist']]
    genie = genieDF[['norm_song', 'norm_artist']]
    flo = floDF[['norm_song', 'norm_artist']]
    vibe = vibeDF[['norm_song', 'norm_artist']]
    bugs = bugsDF[['norm_song', 'norm_artist']]
    apple = appleDF[['norm_song', 'norm_artist']]

    score_column = list(range(100, 0, -1))
    melon.insert(0, 'score', score_column)
    genie.insert(0, 'score', score_column)
    flo.insert(0, 'score', score_column)
    vibe.insert(0, 'score', score_column)
    bugs.insert(0, 'score', score_column)
    apple.insert(0, 'score', score_column)

    melon.loc[:, 'score'] = melon['score'] * 55.0
    genie.loc[:, 'score'] = genie['score'] * 22.9
    flo.loc[:, 'score'] = flo['score'] * 12.6
    vibe.loc[:, 'score'] = vibe['score'] * 11.5
    bugs.loc[:, 'score'] = bugs['score'] * 11.5
    apple.loc[:, 'score'] = apple['score'] * 5.6

    merge6 = merge5[['norm_song', 'norm_artist', 'albumcover']].copy()

    combined = pd.concat([melon, bugs, genie, flo, vibe, apple])

    # 점수 합산을 위한 그룹화
    score_sum = combined.groupby(['norm_song', 'norm_artist'])['score'].sum().reset_index()

    # merge6에 점수를 추가합니다.
    merge6 = pd.merge(merge6, score_sum, on=['norm_song', 'norm_artist'], how='left').fillna(0)

    total = pd.read_csv('total_KR_ENG.csv', delimiter=',', encoding='utf-8', skiprows=1, header=None).T
    total = total.iloc[:,-6:]
    total.columns = ['albumCover', 'songNumber', 'KR_Song', 'KR_artist', 'ENG_Song', 'ENG_artist']
    selected_rows = total.reset_index(drop=True)

    merged_df = pd.concat([merge6, selected_rows], axis=1)


    # 'KR_Song'과 'ENG_Song' 열에서 NaN을 'norm_song' 값으로 대체
    merged_df['KR_Song'] = merged_df['KR_Song'].fillna(merged_df['norm_song'])
    merged_df['ENG_Song'] = merged_df['ENG_Song'].fillna(merged_df['norm_song'])

    # 'KR_artist'와 'ENG_artist' 열에서 NaN을 'norm_artist' 값으로 대체
    merged_df['KR_artist'] = merged_df['KR_artist'].fillna(merged_df['norm_artist'])
    merged_df['ENG_artist'] = merged_df['ENG_artist'].fillna(merged_df['norm_artist'])


    merged_final = merged_df[['norm_song', 'norm_artist', 'albumCover', 'songNumber', 'KR_Song', 'KR_artist', 'ENG_Song', 'ENG_artist', 'score']].reset_index(drop=True)

    start_time = datetime.datetime.now()  # 시작 시간 기록

    # 선택한 열을 출력
    merged_final.to_csv('total.csv', encoding='utf-8-sig')

    finalFIN = merged_final.sort_values('score', ascending=False).reset_index(drop=True)[:100]
    filename = f'total_score_{start_time}.csv'
    finalFIN.to_csv(filename, index=False, encoding='utf-8-sig')

    norm_merged = pd.concat([merge5[['norm_song', 'norm_artist']], merged_final], axis=1)

    melon['Rank'] = pd.Series(range(1, 101))
    bugs['Rank'] = pd.Series(range(1, 101))
    flo['Rank'] = pd.Series(range(1, 101))
    genie['Rank'] = pd.Series(range(1, 101))
    vibe['Rank'] = pd.Series(range(1, 101))
    apple['Rank'] = pd.Series(range(1, 101))

    final = merged_final.sort_values('score', ascending=False).reset_index(drop=True)[:100]


    # DataFrame들을 하나의 DataFrame으로 통합하는 과정
    data_frames = [bugs, melon, genie, vibe, flo, apple]
    names = ['bugs', 'melon', 'genie', 'vibe', 'flo', 'apple']

    # 초기 DataFrame 설정
    final_with_ranks = final.copy()

    final_with_ranks['Rank'] = pd.Series(range(1, 101))


    for df, name in zip(data_frames, names):
        # Merge 작업
        final_with_ranks = pd.merge(final_with_ranks, df[['norm_song', 'norm_artist', 'Rank']], on=['norm_song', 'norm_artist'], how='left', suffixes=('', '_' + name))
        # Merge 후에 새로 생긴 Rank 열의 이름 변경
        final_with_ranks.rename(columns={'Rank_' + name: name}, inplace=True)

    final_with_ranks.fillna(0, inplace=True)

    # 'bugs', 'melon', 'genie', 'vibe', 'flo', 'apple' 열들을 int 타입으로 변환
    columns_to_convert = ['bugs', 'melon', 'genie', 'vibe', 'flo', 'apple']
    final_with_ranks[columns_to_convert] = final_with_ranks[columns_to_convert].astype(int)

    # 선택한 열들만 0을 '-'로 바꾸기
    columns_to_convert = ['bugs', 'melon', 'genie', 'vibe', 'flo', 'apple']
    final_with_ranks[columns_to_convert] = final_with_ranks[columns_to_convert].replace(0, '-')
    final_with_ranks['score'] = final_with_ranks['score'].round(1)
    final_with_ranks['Rank'] = final_with_ranks['score'].rank(ascending=False, method='min').astype(int)
    print('변경 전')
    print(final_with_ranks[final_with_ranks['albumCover'].values ==0])
    merged_df2 = pd.merge(final_with_ranks, melonDF[['norm_song', 'norm_artist', 'albumcover']],
                        on=['norm_song', 'norm_artist'], how='left', suffixes=('', '_melon'))

    # 'albumCover' 값이 0인 경우, melonDF의 'albumcover' 값으로 업데이트합니다.
    merged_df2.loc[merged_df2['albumCover'] == 0, 'albumCover'] = merged_df2['albumcover']

    # 병합 과정에서 생성된 불필요한 열을 제거합니다.
    merged_df2.drop(columns=['albumcover'], inplace=True)

    # 이제 merged_df를 원래의 final_with_ranks DataFrame으로 대체하거나 업데이트합니다.
    final_with_ranks_updated = merged_df2
    print('변경 후')
    print(final_with_ranks_updated[final_with_ranks_updated.albumCover==0])

    merged_df = pd.merge(final_with_ranks_updated, melonDF[['norm_song', 'norm_artist', 'songNum']],
                        on=['norm_song', 'norm_artist'], how='left', suffixes=('', '_melon'))
    # merged_df
    # # 'albumCover' 값이 0인 경우, melonDF의 'albumcover' 값으로 업데이트합니다.
    merged_df.loc[merged_df['songNumber'] == 0, 'songNumber'] = 'm' + merged_df['songNum']



    # 병합 과정에서 생성된 불필요한 열을 제거합니다.
    merged_df.drop(columns=['songNum'], inplace=True)

    # 이제 merged_df를 원래의 final_with_ranks DataFrame으로 대체하거나 업데이트합니다.
    final_with_ranks_updated = merged_df
    final_with_ranks_updated


    current_date = datetime.datetime.now().strftime('%Y-%m-%d')  # '년-월-일' 형식

    filename = f'metachart2_{current_date}.csv'

    final_with_ranks_updated.to_csv(filename, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    start_time = datetime.datetime.now()  # 시작 시간 기록
    print('totalDataFrame')
    print(f"실행 시작 시간: {start_time}")

    main()  # 스크립트의 메인 함수 실행

    end_time = datetime.datetime.now()  # 종료 시간 기록
    print(f"실행 완료 시간: {end_time}")

    # 실행 소요 시간 계산
    duration = end_time - start_time
    print(f"총 실행 시간: {duration}")