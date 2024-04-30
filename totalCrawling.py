import requests
import urllib.request
from bs4 import BeautifulSoup
import csv
import datetime
import re
import time
from bs4 import CData
import lxml

datetime.datetime.today()
now = datetime.datetime.now()
nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')


def 멜론():
    if __name__ == "__main__":
        RANK = 100

        header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Trident/7.0; rv:11.0) like Gecko'}
        req = requests.get('https://www.melon.com/chart/day/index.htm', headers=header)
        html = req.text
        parse = BeautifulSoup(html, 'html.parser')

        titles = parse.find_all("div", {"class": "ellipsis rank01"})
        singers = parse.find_all("div", {"class": "ellipsis rank02"})
        albums = parse.find_all("div", {'class': 'wrap'})

        title = []
        singer = []
        tnumber = []
        snumber = []
        tnumber2 = []
        snumber2 = []
        album = []

        # albums.find('img')

        for a in albums:
            img_tag = a.find('img')
            if img_tag and 'src' in img_tag.attrs:  # img 태그가 존재하고 src 속성이 있는지 확인
                img_url = img_tag['src']
                original_img_url = img_url.split('/melon')[0]  # 원본 이미지 URL 추출
                album.append(original_img_url)

        for t in titles:
            title.append(t.find('a').text)

        for s in singers:
            singer.append(s.find('span', {"class": "checkEllipsis"}).text)

        for n in titles:
            tnumber.append(n.find('a'))
        for t in tnumber:
            a = str(t).split(',')[1][:8]
            tnumber2.append(re.sub(r"\D", "", str(a)))

        for s in singers:
            snumber.append(s.find('span', {"class": "checkEllipsis"}))
        for s in singers:
            s2 = str(s)[77:84]
            snumber2.append(re.sub(r"\D", "", str(s2)))

        with open('c_melon.csv', 'a', encoding='utf-8-sig', newline='') as f:
            melonwriter=csv.writer(f)
            melonwriter.writerow([nowDatetime])
            melonwriter.writerows([title,singer,tnumber2,snumber2,album])
            f.close()


bugsdata = requests.get('https://music.bugs.co.kr/chart/track/day/total')
bugssoup = BeautifulSoup(bugsdata.text, 'html.parser')


def 벅스():
    if __name__ == "__main__":

        bugstitles = bugssoup.find_all("p", {"class": "title"})
        bugssingers = bugssoup.find_all("p", {"class": "artist"})
        bugsalbums = bugssoup.find_all('tr')

        bugstitle = []
        bugssinger = []

        bugsalbum = []

        for album in bugsalbums:
            img_tag = album.find('img')
            # print(img_tag)
            if img_tag and 'src' in img_tag.attrs:  # img 태그가 존재하고 src 속성이 있는지 확인
                img_url = img_tag['src']
                # print(img_url)
                img_url_original = img_url.split('?')[0].replace('/50/', '/original/')
                # print(img_url_original)
                bugsalbum.append(img_url_original)
        # bugsalbum[:-1]

        for bugst in bugstitles:
            bugstitle.append(bugst.find('a').text)

        for bugss in bugssingers:
            bugssinger.append(bugss.find('a').text)


        with open('c_bugs.csv', 'a', encoding='utf-8-sig', newline='') as f:
            bugswriter = csv.writer(f)
            bugswriter.writerow([nowDatetime])
            bugswriter.writerows([bugstitle, bugssinger, bugsalbum[:-1]])
            f.close()


def 플로():
    if __name__ == "__main__":

        RANK = 100
        floreq = requests.get('https://www.music-flo.com/api/meta/v1/chart/track/1?timestamp=1610849102811')
        flodata = floreq.json()

        flo_musics = flodata['data']['trackList']

        flotitle = []
        flosinger = []
        floalbum = []

        for floa in flo_musics:
            floalbum.append(floa['album']['img']['urlFormat'])
        for flot in flo_musics:
            flotitle.append(flot['name'])
        for flos in flo_musics:
            flosinger.append(flos['artistList'][0]['name'])

        with open('c_flo.csv', 'a', encoding='utf-8-sig', newline='') as f:
            flowriter = csv.writer(f)
            flowriter.writerow([nowDatetime])
            flowriter.writerows([flotitle, flosinger,floalbum])
            f.close()


def 지니():
    if __name__ == "__main__":
        RANK = 50

        header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Trident/7.0; rv:11.0) like Gecko'}

        gereq = requests.get('https://genie.co.kr/chart/top200?ditc=D&rtm=N', headers=header)
        gereq2 = requests.get('https://genie.co.kr/chart/top200?ditc=D&ymd=19000101&hh=17&rtm=N&pg=2', headers=header)

        gehtml = gereq.text
        gehtml2 = gereq2.text
        geparse = BeautifulSoup(gehtml, 'html.parser')
        geparse2 = BeautifulSoup(gehtml2, 'html.parser')

        # gealbums = gereq.find_all('td')

        getitles = geparse.find_all("td", {"class": "info"})
        getitles2 = geparse2.find_all("td", {"class": "info"})
        gesingers = geparse.find_all("td", {"class": "info"})
        gesingers2 = geparse2.find_all("td", {"class": "info"})

        gealbums = geparse.find_all('a', {'class':'cover'})
        gealbums2 = geparse2.find_all('a', {'class':'cover'})


        getitle = []
        gesinger = []
        gealbum = []

        # gealbums
        for gea in gealbums:
            img_tag = gea.find('img')
            # print(img_tag)
            if img_tag and 'src' in img_tag.attrs:  # img 태그가 존재하고 src 속성이 있는지 확인
                img_url = img_tag['src']
                # print(img_url)
                img_url_original = 'http:' + img_url.split('/dims')[0].replace('140x140', '600x600')
                # print(img_url_original)
                gealbum.append(img_url_original)

        for gea2 in gealbums2:
            img_tag = gea2.find('img')
            # print(img_tag)
            if img_tag and 'src' in img_tag.attrs:  # img 태그가 존재하고 src 속성이 있는지 확인
                img_url = img_tag['src']
                # print(img_url)
                img_url_original = 'http:' + img_url.split('/dims')[0].replace('140x140', '600x600')
                # print(img_url_original)
                gealbum.append(img_url_original)

        for get in getitles:
            getitle.append(get.find('a').text.strip())

        for get2 in getitles2:
            getitle.append(get2.find('a').text.strip())

        for ges in gesingers:
            gesinger.append(ges.find("a", {"class": "artist ellipsis"}).text.strip())

        for ges2 in gesingers2:
            gesinger.append(ges2.find("a", {"class": "artist ellipsis"}).text.strip())

        with open('c_genie.csv', 'a', encoding='utf-8-sig', newline='') as f:
            gewriter = csv.writer(f)
            gewriter.writerow([nowDatetime])
            gewriter.writerows([getitle, gesinger, gealbum])
            f.close()


def 바이브():
    if __name__ == "__main__":
        RANK = 100

        vibeheader = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Trident/7.0; rv:11.0) like Gecko'}
        vibereq = requests.get('https://apis.naver.com/vibeWeb/musicapiweb/vibe/v1/chart/track/total',
                                headers=vibeheader)
        vibehtml = vibereq.text
        vibeparse = BeautifulSoup(vibehtml, 'lxml-xml')

        vibetitles = vibeparse.find_all('trackTitle')
        vibeartists = vibeparse.find_all('album')

        vibetitle = []
        vibeartist = []
        vibealbum = []

        for vibeal in vibeartists:
            vibea = vibeal.find('imageUrl')
            vibealbum.append(vibea.text.split('?type')[0])

        for vibet in vibetitles:
            vibetitle.append(vibet.text)

        for vibea in vibeartists:
            vibeartist.append(vibea.find('artistName').text)


        with open('c_vibe.csv', 'a', encoding='utf-8-sig', newline='') as f:
            vibewriter = csv.writer(f)
            vibewriter.writerow([nowDatetime])
            vibewriter.writerows([vibetitle])
            vibewriter.writerows([vibeartist])
            vibewriter.writerows([vibealbum])

            f.close()


def 애플():
    if __name__ == "__main__":
        appledata = requests.get(
            'https://music.apple.com/kr/playlist/%EC%98%A4%EB%8A%98%EC%9D%98-top-100-%EB%8C%80%ED%95%9C%EB%AF%BC%EA%B5%AD/pl.d3d10c32fbc540b38e266367dc8cb00c')
        applesoup = BeautifulSoup(appledata.content.decode('utf-8-sig', 'replace'), 'html.parser')
        a = applesoup.find_all('meta', {'property': 'music:song'})

        pattern = r'\d+'
        applesongList = []

        for b in a:
            applesongList.append(f"{str(b).split('/')[5]}/{re.findall(pattern, str(b).split('/')[6])[0]}")

        count = 1
        patternArtist = r'(?<= - )(.*?)(?= - )'
        patternSong = r'^.*?(?=\s-\s)'

        pattern_artistUK = r'(?<=– Song by )[^–]+'
        pattern_songUK = r'(?<=‎)[^–]+'

        artistAppleListKR = []
        songAppleListKR = []
        albumCover = []


        for a in applesongList:
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
                    pass

        artistAppleListUK = []
        songAppleListUK = []

        for a in applesongList:
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
                    count +=1

                except AttributeError: pass


        with open('c_apple.csv', 'a', encoding='utf-8-sig', newline='') as f:
            applewriter = csv.writer(f)
            applewriter.writerow([nowDatetime])
            applewriter.writerows([songAppleListKR])
            applewriter.writerows([artistAppleListKR])
            applewriter.writerows([songAppleListUK])
            applewriter.writerows([artistAppleListUK])
            applewriter.writerows([albumCover])


def main():
    멜론()
    print('멜론 끝')
    벅스()
    print('벅스 끝')
    플로()
    print('플로 끝')
    지니()
    print('지니 끝')
    바이브()
    print('바이브 끝')
    애플()
    print('애플 끝')



if __name__ == "__main__":
    start_time = datetime.datetime.now()  # 시작 시간 기록
    print('totalCrawling')
    print(f"실행 시작 시간: {start_time}")

    main()  # 스크립트의 메인 함수 실행

    end_time = datetime.datetime.now()  # 종료 시간 기록
    print(f"실행 완료 시간: {end_time}")

    # 실행 소요 시간 계산
    duration = end_time - start_time
    print(f"총 실행 시간: {duration}")