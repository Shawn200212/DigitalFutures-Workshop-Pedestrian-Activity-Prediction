import math
import os, json
from PIL import Image
from io import BytesIO
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests, csv
import logging
from fake_useragent import UserAgent
import re
from typing import Literal, Union

from .ctr import bdmc2wgs
from .wgs2bd import wgs2bdmc
from .time_selector import TimeFuncs



def Random_choose_useragent():
    ua = UserAgent()
    headers = {
        'User-Agent': ua.random}
    return headers

def dist(x1,y1, x2,y2):
    return ((x2-x1)**2+(y2-y1)**2)**0.5

@dataclass
class StreetviewData:
    image: Image
    # Bug修复：将 str or int or None 替换为 Union[str, int, None]
    date: Union[str, int, None]
    sid: Union[str, int, None]
    udt_sid: Union[str, int, None]
    direction: Union[str, int, None]
    success: bool

class StreetViewScrawler:
    def __init__(self,
                 date_selector: callable = TimeFuncs.Latest,
                 img_date: str = '',
                 z_level: int = 3,
                 flat_img_size: int = 512,
                 max_distance_thre: int = 100,
                 multi_thread: bool = True,
                 ):
        """
        :param date_selector: 确定图像日期的函数，可从bdcrawler.TimeFuncs中选择，也可以自定义。自定义方式请参考源码
        :param img_date:      仅当date_selector为TimeFuncs.Nearest时生效（或自定义函数）
        :param z_level:       全景图的缩放等级，最小为2，z_level越大，图像越大。抓取四方向图时不生效。
        :param flat_img_size: 抓取四方向图时，图像大小为(flat_img_size, flat_img_size) 
        :param max_distance_thre: 抓取时，若传入点与街景图像真实点距离超过该阈值，则放弃抓取
        """
        self.headers = Random_choose_useragent()
        self.date_selector = date_selector
        if date_selector == TimeFuncs.Nearest and img_date is None:
            raise RuntimeError("img_date must be provided when date_selector is Nearest")
        self.img_date = str(img_date)
        assert z_level>1, ""
        self.z = z_level
        self.tiles = None

        self.y_tiles = None
        self.x_tiles = None
        self.get_tiles()

        # assert flat_img_size in [256, 512, 1024], "invalid image size for flat images"
        
        self.flat_img_size = flat_img_size
        self.max_distance_thre = max_distance_thre
        self.multi_thread = multi_thread
        self.num_workers = 4


    def get_tiles(self):
        self.y_tiles = 2**(self.z - 2)
        self.x_tiles = self.y_tiles * 2
        self.tiles = []
        for i in range(self.y_tiles):
            for j in range(self.x_tiles):
                self.tiles.append(f'{i}_{j}')
        assert self.x_tiles*self.y_tiles == len(self.tiles)




    def xy2sid(self, bdx, bdy):
        """x,y to sid
        由百度坐标得到街景的sid
        sid可以唯一标识特定时间的街景点
        """
        params = {
            "action": 0,
            'x': bdx,
            'y': bdy,
            'l': 18.367179030452565,
            "mode": 'day',
            "auth": 'ySSM9eGPF3F7Gv0%2540zBEKaKxRbMdRXP%253D9uxHHBzELTRxt1qo6DF%253D%253DCy1uVt1GgvPUDZYOYIZuVtcvY1SGpuEt2gz4yExIff%253DfxXwPWv3GuEtXzljPaVjyBDEHKOQUWYxcEWe1GD8zv7u%2540ZPuVteuVtegvcguxHHBzELTLTtmqQQD%253DCrZZWuz',
            't': 1553246985040,
            'fn': 'jsonp69972182'
        }

        # try:
        r = requests.get("https://mapsv0.bdimg.com/?qt=qsdata", params, headers = self.headers, timeout = (3,7))
        # except Exception as e:
        #     print("x,y to sid失败！")
        #     print(str(e))
        #     return -1, 'error'
        # else:
        str1 = str(r.content, encoding = "utf8")
        jsonstr = str1.split('(')[1].split(')')[0]
        j = json.loads(jsonstr)
        if j["result"]["error"] == 0:
            sid = j['content']['id']
            road_x, road_y = j['content']['x']/100, j['content']['y']/100
            if dist(bdx, bdy, road_x, road_y)<self.max_distance_thre:
                return sid, 'success' # 距离太远的会失真
            else:
                print(dist(bdx, bdy, road_x, road_y), end=' ')
                return -1, 'too far'
        else:
            return -1, 'no image'

    def sid2all_stv_info(self, sid) -> dict:
        params2 = {
            'sid': sid,
            'pc':1,
            'fn': 'jsonp.p3991630'
        }
        infos = {}
        try:
            r = requests.get("https://mapsv0.bdimg.com/?qt=sdata", params2, headers = self.headers, timeout = (3,7))
        except Exception as e:
            print("sid to img失败！")
            print(str(e))
        else:
            str1 = str(r.content, encoding = "utf8")
            p2 = re.compile(r'[(](.*)[)]', re.S)  # 贪婪匹配
            jsonstr2 = str(re.findall(p2, str1)[0])

            j = json.loads(jsonstr2)
            content = j['content'][0]
            out_dict = {'time_lines': content['TimeLine'],
                        'bdx': content['RX'],
                        'bdy': content['RY'],
                        'move_dir': content['MoveDir']}
            return out_dict
            # direction = float(j['content'][0]['MoveDir'])   # 获取道路的方位
            # timelines =  j['content'][0]['TimeLine']
            # bdx, bdy = j['content'][0]['RX'], j['content'][0]['RY']
            #
            # # print([timeline['TimeLine'] for timeline in timelines])
            # timestamp = self.date_selector(timelines, self.img_date)
            # infos['dir'] = direction
            # infos['time'] = timestamp['TimeLine']
            # infos['udtSID'] = timestamp['ID']
            # infos['bdx'] = bdx
            # infos['bdy'] = bdy
            # bd_lon, bd_lat = bdmc2wgs(bdx/100, bdy/100)
            # infos['bd_lon'],infos['bd_lat'] = bd_lon, bd_lat
            # return infos

    def sid2stv_info(self, sid):
        params2 = {
            'sid': sid,
            'pc':1,
            'fn': 'jsonp.p3991630'
        }
        infos = {}
        try:
            r = requests.get("https://mapsv0.bdimg.com/?qt=sdata", params2, headers = self.headers, timeout = (3,7))
        except Exception as e:
            print("sid to img失败！")
            print(str(e))
        else:
            str1 = str(r.content, encoding = "utf8")
            p2 = re.compile(r'[(](.*)[)]', re.S)  # 贪婪匹配
            jsonstr2 = str(re.findall(p2, str1)[0])

            j = json.loads(jsonstr2)
            direction = float(j['content'][0]['MoveDir'])   # 获取道路的方位
            timelines =  j['content'][0]['TimeLine']
            bdx, bdy = j['content'][0]['RX'], j['content'][0]['RY']

            # print([timeline['TimeLine'] for timeline in timelines])
            timestamp = self.date_selector(timelines, self.img_date)
            if timestamp == []:
                return None
            infos['dir'] = direction
            infos['time'] = timestamp['TimeLine']
            infos['udtSID'] = timestamp['ID']
            infos['bdx'] = bdx
            infos['bdy'] = bdy
            bd_lon, bd_lat = bdmc2wgs(bdx/100, bdy/100)
            infos['bd_lon'],infos['bd_lat'] = bd_lon, bd_lat
            return infos

    def fetch_single_tile(self, udt_sid: str, tile: str):
        y_pos, x_pos = list(map(int, tile.split("_")))
        params3 = {
            'sid': udt_sid,
            'pos': tile,
            'z': self.z,
        }
        r = requests.get("https://mapsv1.bdimg.com/?qt=pdata", params3, headers=self.headers, timeout=(3, 7))
        return r, x_pos, y_pos

    def stv_info2panorama(self, udt_sid: str):
        panorama = Image.new("RGB", (self.x_tiles * 512, self.y_tiles * 512))
        if self.multi_thread == False:
            for tile in self.tiles:
                r, x_pos, y_pos = self.fetch_single_tile(udt_sid, tile)
                if r.headers['Content-Type'] == 'image/jpeg' or 'text':
                    tile_image = Image.open(BytesIO(r.content))
                    panorama.paste(im=tile_image, box=(x_pos * 512, y_pos * 512))
            return
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 先提交所有任务，获取完之后再paste
            all_futures = []
            for tile in self.tiles:
                future = executor.submit(self.fetch_single_tile, udt_sid, tile)
                all_futures.append(future)
            for future in as_completed(all_futures):
                r, x_pos, y_pos = future.result()
                if r.headers['Content-Type'] == 'image/jpeg' or 'text':
                    tile_image = Image.open(BytesIO(r.content))
                    panorama.paste(im=tile_image, box=(x_pos * 512, y_pos * 512))
        return panorama



    def stv_info2image(self, udt_sid: str, udt_dir:float, heading, fovy: int = 90):
        param3 = {
            'fovy': fovy,
            'quality': 100,
            'panoid': udt_sid,  # panoid 与sid对应
            'heading': (heading + udt_dir) % 360,
            'width': self.flat_img_size,
            'height': self.flat_img_size,
            }
        r = requests.get("https://mapsv0.bdimg.com/?qt=pr3d", param3, headers=self.headers, timeout=(3, 7))
        if r.headers['Content-Type'] == 'image/jpeg':
            tile_image = Image.open(BytesIO(r.content))
            return tile_image
        else:
            raise RuntimeError('get tile image error')



    def get_sid(self, lon: float, lat: float, n_tries: int = 3) -> str:
        bdx, bdy = wgs2bdmc(lon, lat)
        try_count = 1
        while try_count <= n_tries:
            try:
                sid, sid_info = self.xy2sid(bdx, bdy)
                return sid
            except Exception as e:
                try_count += 1
                print(f'{lon}, {lat}, try {try_count}, err {str(e)}')
        else:
            print(f'{lon}, {lat}, failed after {n_tries} tries')
            return '-1' # failed

    def get_stv_info(self, lon: float, lat: float, n_tries: int = 3) -> dict:
        bdx, bdy = wgs2bdmc(lon, lat)
        try_count = 1

        while try_count <= n_tries:
            try:
                sid, sid_info = self.xy2sid(bdx, bdy)
                if sid != -1:
                    stv_info = self.sid2all_stv_info(sid)
                    return stv_info
                else:
                    print(sid_info)
                    return {}
            except Exception as e:
                try_count += 1
                print(f'{lon}, {lat}, try {try_count}, err {str(e)}')
        else:
            print(f'{lon}, {lat}, failed after {n_tries} tries')
            return {}


    def get_flatimage(self, lon: float, lat: float, direction: int, n_tries: int = 3) -> StreetviewData:
        return self.get_image(lon, lat, 'flat', direction, n_tries)

    def get_panorama(self, lon: float, lat: float, n_tries: int = 3) -> StreetviewData:
        return self.get_image(lon, lat, 'panorama', None, n_tries)


    def get_image(self, lon: float, lat: float, image_type: str, direction: int or None, n_tries: int = 3) -> StreetviewData:
        bdx, bdy = wgs2bdmc(lon, lat)
        try_count = 1

        while try_count <= n_tries:
            try:
                sid, sid_info = self.xy2sid(bdx, bdy)
                if sid != -1:
                    stv_info = self.sid2stv_info(sid)
                    if stv_info == None:
                        return StreetviewData(None, None, sid, -1, direction, False)
                    if image_type == 'flat':
                        date = stv_info['time']
                        image = self.stv_info2image(stv_info['udtSID'], stv_info['dir'], direction)
                    elif image_type == 'panorama':
                        date, direction = stv_info['time'], stv_info['dir']
                        image = self.stv_info2panorama(stv_info['udtSID'])
                    else:
                        raise ValueError('invalid image type')
                    return StreetviewData(image, date, sid, stv_info['udtSID'], direction, True)
                else:
                    print(sid_info)
                    return StreetviewData(None, None, sid, -1, direction, False)
            except Exception as e:
                try_count += 1
                print(f'{lon}, {lat}, try {try_count}, err {str(e)}')
        else:
            print(f'{lon}, {lat}, failed after {n_tries} tries')
            return StreetviewData(None, None, -1, -1, direction, False)




    def udt_id2flatimage(self, udt_id: str, udt_dir: float, direction: int, n_tries: int = 3) -> StreetviewData:
        return self.udt_id2image(udt_id, 'flat', udt_dir, direction, n_tries)

    def udt_id2panorama(self, udt_id: str, n_tries: int = 3) -> StreetviewData:
        return self.udt_id2image(udt_id, 'panorama', None,None, n_tries)

    def udt_id2image(self, udt_id: str, image_type: str, udt_dir: float or None, direction: int or None, n_tries: int = 3) -> StreetviewData:
        try_count = 1
        while try_count <= n_tries:
            try:
                stv_info = self.sid2stv_info(udt_id)
                if image_type == 'flat':
                    image = self.stv_info2image(udt_id, udt_dir, direction)
                elif image_type == 'panorama':
                    image = self.stv_info2panorama(udt_id)
                else:
                    raise ValueError('invalid image type')
                return StreetviewData(image, None, None, udt_id, direction, True)
            except Exception as e:
                try_count += 1
                print(f'{udt_id}, try {try_count}, err {str(e)}')
        else:
            print(f'{udt_id}, failed after {n_tries} tries')
            return StreetviewData(None, None, None, udt_id, direction, False)

    # def udtID2image(self, udtID: str, image_type: str, direction: int or None, n_tries: int = 3) -> StreetviewData:
    #     try_count = 1
    #     while try_count <= n_tries:
    #         try:
    #             stv_info = self.sid2stv_info(udtID)
    #             if image_type == 'flat':
    #                 image, date = self.stv_info2image(stv_info, direction)
    #             elif image_type == 'panorama':
    #                 image, date, direction = self.stv_info2panorama(stv_info)
    #             else:
    #                 raise ValueError('invalid image type')
    #             return StreetviewData(image, date, udtID, direction, True)
    #         except Exception as e:
    #             try_count += 1
    #             print(f'{udtID}, try {try_count}, err {str(e)}')
    #     else:
    #         print(f'{udtID}, failed after {n_tries} tries')
    #         return StreetviewData(None, None, udtID, direction, False)

