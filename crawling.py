from requests import get
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from time import sleep
import os
from tkinter import Tk
from tkinter.simpledialog import askstring
from shutil import copyfileobj
from base64 import b64decode

window = Tk()
# 유저가 입력한 키워드들을 리스트로 저장
keyword_list = askstring(
    title='구글 이미지 다운로더', prompt='검색할 키워드를 콤마(,)로 구분해주세요(키워드가 두단어 이상이라면 공백으로 구분하세요').split(',')
window.destroy()
PATH = './chromedriver'
# 크롬을 띄우지 않는 옵션 설정
options = webdriver.ChromeOptions()
options.headless = True
try:
    driver = webdriver.Chrome(PATH, options=options)
    # 키워드 별로 반복문을 돌림
    for keyword in keyword_list:
        # 구글의 키워드 검색결과의 이미지 탭으로 가는 링크
        url = '''https://www.google.com/search?tbm=isch&source=hp&biw=&bih=&ei=0VgdX4viLcq9hwOB7IngCQ&q=''' + keyword.strip()
        driver.get(url)
        # 폴더가 없으면 키워드로 폴더를 만듦
        try:
            parent_dir = "./images/"
            path = os.path.join(parent_dir, keyword)
            os.mkdir(path)
            print('Path made for ' + keyword)
        except:
            print('Directory already exists for keyword:', keyword)
        # 페이지가 로딩이 느릴 때를 대비해 5초를 기다림
        sleep(5)
        # 다음 페이지의 가장 밑까지 가게한다
        last_height = driver.execute_script('return document.body.scrollHeight')
        while True:
            driver.execute_script(
                'window.scrollTo(0,document.body.scrollHeight)')
            sleep(1)
            new_height = driver.execute_script(
                'return document.body.scrollHeight')
            if new_height == last_height:
                # 이미지를 더 로딩하는 버튼을 누름
                try:
                    driver.find_element_by_class_name('mye4qd').click()
                    sleep(1)
                except:
                    break
            last_height = new_height
        print('Done scrolling')
        # 페이지 내 모든 이미지를 찾음
        image_urls = driver.find_elements_by_css_selector('img.rg_i.Q4LuWd')
        print('Found', len(image_urls), 'results')
        count = 1
        for image_thumbnail in image_urls:
            # 원본 이미지를 다운로드하기 위해 썸네일을 클릭함
            ActionChains(driver).click(image_thumbnail).perform()
            sleep(2)
            # 같은 클래스를 가진 이미지가 여러개이기 때문에 xpath를 통해 지정함
            try:
                image_url = driver.find_element_by_xpath(
                '/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div/div[2]/a/img').get_attribute("src")
            except:
                continue

            with open(f'./images/{keyword}/{count}.jpg', 'wb') as image_file:
                # 이미지가 base64로 인코딩된 텍스트일때
                if image_url.startswith('data:image/jpeg;base64,'):
                    # 이미지 데이터를 디코딩해 저장한다(데이터를 해석해 저장)-23번째 인덱스부터 시작한다.
                    image_file.write(b64decode(image_url[23:]))
                else:
                    # requests.get 함수를 사용해 링크에서 이미지(와 기타 등등)을 가져온다-stream은 안정성을 위해 추가
                    try:
                        response = get(image_url, stream=True)
                        # 이미지 데이터를 열고있는 파일에 저장한다.
                        copyfileobj(response.raw, image_file)
                        # 버퍼를 없앤다
                        del response
                    except:
                        print(count)
                        pass
                count += 1
        print('Done scrapping images')
finally:
    # 브라우저를 닫고 더 이상의 셀레니움을 사용한 코드 실행을 멈춤
    driver.quit()