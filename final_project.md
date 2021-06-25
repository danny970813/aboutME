# 專題製作_M10915031_鄭善謙
1. 作品名稱
    
    人流計數於疫情下的應用
    
2. 摘要說明

   在疫情之下，我們的生活方式被迫改變，人與人之間必須盡可能地減少面對面的接觸，而人潮眾多的場所，也成為了可能讓病毒滋長的溫床，所以本次研究以人流計數的方式去判斷公共場所是否安全，透過物件追蹤的方法快速取得疑似案例的接觸者，希望能用自動化的方式協助公共場所的防疫工作。


3. 系統簡介

    3.1 創作發想

    COVID-19已經從2019年末肆虐至今，而台灣很幸運地在之前擋下了好幾波可能擴散的危機，但我們的防線最終還是被病毒給攻破，從5月開始台灣便進入了社區感染的階段，而在疫情爆發後，各大賣場都有發生搶購物資的現象，但其實這也是一種變相的群聚發生，在這個時候如果我們還加入購買物資的行列，反而是將自己置身在危險當中，因此本次研究以人流計算為目的，以深度學習的方式去對影片人流做計算，自動的依人流多寡分辨該場所安全與否，以此避免非必要的群聚，保護自己的同時也保護他人。


    3.2 硬體架構
    
    硬體的部分是使用i5-7500的CPU進行測試，在訓練時使用GTX-1080ti加速訓練的進行。
    

    |     CPU      |  i5-7500  |
    |:------------:|:---------:|
    |     GPU      | GTX1080ti |
    |     RAM      |    16G    |
    | AI Framework |  pytorch  |


    3.3 工作原理及流程
    
    ![](https://i.imgur.com/BwL0m6e.png)

    如上圖，首先透過JDE的方式對每一幀的行人位置進行預測，並產生Embedding的結果，之後對過卡爾曼濾波器預測前一幀的行人在這一幀可能的位置，在將兩者交由匈牙利演算法進行匹配，匹配成功就會賦予之前相同的id，反之則賦予一個新id，最後輸出整段影片的追蹤結果，再透過計算每一幀預測的行人數量達到人流計數的效果。

    3.4 資料集建立方式
    
    直接以MOT-17進行訓練，MOT-17是一個適合用來進行物件追蹤的資料集，其中包含14個時長18秒到1分鐘左右的影片，包含7個不同場景，平均每幀包括20人左右。以下為資料集中ground truth的標註內容。
    
    ![](https://i.imgur.com/zVv1H0y.jpg)

    ground truth內容由左至右依序為:

    |  |  |  |  |  |  |  |  |  |
    |:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
    | Frame number | Identity number | Bounding box-left | Bounding box-top | Bounding box-top-width | Bounding box-top-height | Confidence score | Class | Visibility|

    Class總共有以下12個類別:

    |         Class          | ID  |
    |:----------------------:|:---:|
    |       Pedestrian       |  1  |
    |   Person on vehicle    |  2  |
    |          Car           |  3  |
    |        Bicycle         |  4  |
    |       Motorbike        |  5  |
    | Non motorized vehicle  |  6  |
    |     Static person      |  7  |
    |       Distractor       |  8  |
    |        Occluder        |  9  |
    | Occluder on the ground | 10  |
    |     Occluder full      | 11  |
    |       Reflection       | 12  |

    3.5 模型選用與訓練

    model的選用是參考Towards Real-Time Multi-Object Tracking中的JDE方法，JDE是在SDE的基礎之上改進而來，SDE是屬於tracking by detection的two-stage方法，透過將Detector產生的特徵圖送入Embedding model後得到輸出的Embedding再交由卡爾曼濾波器進行預測，以及匈牙利演算法進行匹配。而JDE為了加快執行速度，則是將Detector和Embedding model整合在一起成了one-stage的方法，由一個model同時預測類別、位置和Embedding結果。
    
    ![](https://i.imgur.com/cew1Fwc.jpg)
    
    以下為JDE的網路架構:
    
    ![](https://i.imgur.com/nKazrtL.jpg)
    
    上半的部分為由Darknet53提取特徵圖，並交由FPN產生三個不同尺度的特徵圖，這也就是YOLOv3，後續由JDE的prediction head進行預測，最後輸出預測的類別、位置和Embedding結果。之後和SDE相同，透過卡爾曼濾波器和匈牙利演算法去匹配每個id，最後輸出整段影片的追蹤結果。
  

4. 實驗結果

    要以人流進行場所安全的判定需要先設定一個標準，因此參考「第三級疫情警戒：停止室內5人、室外10人以上的聚會」的標準，在安全判定上設定為：


    |   safe    | 低於5人  |
    |:---------:|:--------:|
    |  careful  | 5至10人  |
    | dangerous | 超過10人 |




    4.1 測試與比較

    以下為菜市場的監視器畫面片段，可以看到在路中間的行人都有被追蹤到，但左下角被汽車所遮擋只露出上半身的行人沒有被偵測到，而在影片期間人數保持在4~8人之間，因此被判定介於安全到須注意之間。
    
    ![](https://i.imgur.com/sMqoQWf.gif)
    
    以下為youtube上日本澀谷路口的一個監視器畫面片段，這個場景特別的地方在於拍攝時間屬於晚上，所以畫面充斥著不同的光源，並且畫面中央有一個路燈和兩個路標，在行人穿越馬路時會被這些遮蔽物所阻擋，所以可以看到id為1的行人在穿越馬路時被路標所檔住後無法被追蹤到，因此賦予它一個新的id，但這對計數其實是沒有太大的影響的，它只是將賦予id的對象弄錯，但偵測總數不會因此而改變。而在影片期間人數保持在20人以上，因此被判定為危險。
    
    ![](https://i.imgur.com/DTs5vaj.gif)

    4.2 追蹤的其它應用

    使用物間追蹤的方式進行人群計數最大的優點在於我們可以獲得每個物件的移動資訊，因此我們可以拿來做一些其他的應用，以下為假定要匡列某個確診者的接觸對象，假定紅框id為2的行人為確診者，因為病毒是有淺伏期的，不會馬上輻射狀的感染，所以我們以距離為判斷依據，當有其他行人接近id為2號的行人時，我們便將它標註為黃框，並追蹤它之後的移動位置，如果後續能搭配人臉辨識系統將有助於加速疫調，能夠快速匡列出風險較高的目標。
    
    ![](https://imgur.com/7smLs95.gif)
    

5. 結論

    使用物件追蹤的方式進行人流計數可以賦予不同幀的同一物件一個單獨id，能夠減少每一幀重複計算的機會。但是在遮蔽物較多或物件重疊發生頻繁的狀況下，容易將同一物件當作不同物件賦予一個新的id，造成追蹤錯誤的狀況發生。


6. 參考文獻

    [Towards Real-Time Multi-Object Tracking](https://arxiv.org/abs/1909.12605)
       
    
7. 附錄

    7.1 colab源碼

    [colab](https://colab.research.google.com/drive/1YFMrvv_7tQTc-cexOK0crsIEMAV71t_K?usp=sharing)

    7.2 資料集及標註黨
    
    [MOT-17](https://motchallenge.net/data/MOT17/)