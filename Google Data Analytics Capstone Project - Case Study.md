```python
import numpy as np
import pandas as pd 
```


```python
df = pd.read_csv('final_tripdata.csv')
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ride_id</th>
      <th>rideable_type</th>
      <th>started_at</th>
      <th>ended_at</th>
      <th>start_station_name</th>
      <th>start_station_id</th>
      <th>end_station_name</th>
      <th>end_station_id</th>
      <th>start_lat</th>
      <th>start_lng</th>
      <th>end_lat</th>
      <th>end_lng</th>
      <th>member_casual</th>
      <th>day_week_start</th>
      <th>duration</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000001004784CD35</td>
      <td>docked_bike</td>
      <td>2020-07-22 15:38:23</td>
      <td>2020-07-22 15:56:47</td>
      <td>Wolcott (Ravenswood) Ave &amp; Montrose Ave</td>
      <td>238</td>
      <td>Southport Ave &amp; Clybourn Ave</td>
      <td>307</td>
      <td>41.961406</td>
      <td>-87.676169</td>
      <td>41.920771</td>
      <td>-87.663712</td>
      <td>casual</td>
      <td>4.0</td>
      <td>1824.0</td>
      <td>42501.53</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000002EBE159AE82</td>
      <td>electric_bike</td>
      <td>2021-06-22 17:25:15</td>
      <td>2021-06-22 17:31:34</td>
      <td>Clinton St &amp; Jackson Blvd</td>
      <td>638</td>
      <td>Milwaukee Ave &amp; Grand Ave</td>
      <td>13033</td>
      <td>41.878261</td>
      <td>-87.641155</td>
      <td>41.891495</td>
      <td>-87.648179</td>
      <td>member</td>
      <td>3.0</td>
      <td>619.0</td>
      <td>14982.48</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00000550C665101A</td>
      <td>docked_bike</td>
      <td>2020-06-06 15:20:01</td>
      <td>2020-06-06 16:28:09</td>
      <td>Sheffield Ave &amp; Waveland Ave</td>
      <td>114</td>
      <td>Kedzie Ave &amp; Milwaukee Ave</td>
      <td>260</td>
      <td>41.949399</td>
      <td>-87.654529</td>
      <td>41.929567</td>
      <td>-87.707857</td>
      <td>casual</td>
      <td>7.0</td>
      <td>10808.0</td>
      <td>56896.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0000127970C84F62</td>
      <td>docked_bike</td>
      <td>2020-05-30 06:36:36</td>
      <td>2020-05-30 06:55:28</td>
      <td>Green St &amp; Madison St</td>
      <td>198</td>
      <td>Wells St &amp; Concord Ln</td>
      <td>289</td>
      <td>41.881900</td>
      <td>-87.648800</td>
      <td>41.912100</td>
      <td>-87.634700</td>
      <td>member</td>
      <td>7.0</td>
      <td>1892.0</td>
      <td>33329.42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00001A81D056B01B</td>
      <td>classic_bike</td>
      <td>2021-04-14 08:10:11</td>
      <td>2021-04-14 08:19:14</td>
      <td>Wood St &amp; Hubbard St</td>
      <td>13432</td>
      <td>Morgan St &amp; Lake St</td>
      <td>TA1306000015</td>
      <td>41.889899</td>
      <td>-87.671473</td>
      <td>41.885483</td>
      <td>-87.652305</td>
      <td>member</td>
      <td>4.0</td>
      <td>903.0</td>
      <td>19670.18</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dtypes
```




    ride_id                object
    rideable_type          object
    started_at             object
    ended_at               object
    start_station_name     object
    start_station_id       object
    end_station_name       object
    end_station_id         object
    start_lat             float64
    start_lng             float64
    end_lat               float64
    end_lng               float64
    member_casual          object
    day_week_start        float64
    duration              float64
    distance              float64
    dtype: object




```python
df.shape 
```




    (5515094, 16)




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ride_id</th>
      <th>rideable_type</th>
      <th>started_at</th>
      <th>ended_at</th>
      <th>start_station_name</th>
      <th>start_station_id</th>
      <th>end_station_name</th>
      <th>end_station_id</th>
      <th>start_lat</th>
      <th>start_lng</th>
      <th>end_lat</th>
      <th>end_lng</th>
      <th>member_casual</th>
      <th>day_week_start</th>
      <th>duration</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000001004784CD35</td>
      <td>docked_bike</td>
      <td>2020-07-22 15:38:23</td>
      <td>2020-07-22 15:56:47</td>
      <td>Wolcott (Ravenswood) Ave &amp; Montrose Ave</td>
      <td>238</td>
      <td>Southport Ave &amp; Clybourn Ave</td>
      <td>307</td>
      <td>41.961406</td>
      <td>-87.676169</td>
      <td>41.920771</td>
      <td>-87.663712</td>
      <td>casual</td>
      <td>4.0</td>
      <td>1824.0</td>
      <td>42501.53</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000002EBE159AE82</td>
      <td>electric_bike</td>
      <td>2021-06-22 17:25:15</td>
      <td>2021-06-22 17:31:34</td>
      <td>Clinton St &amp; Jackson Blvd</td>
      <td>638</td>
      <td>Milwaukee Ave &amp; Grand Ave</td>
      <td>13033</td>
      <td>41.878261</td>
      <td>-87.641155</td>
      <td>41.891495</td>
      <td>-87.648179</td>
      <td>member</td>
      <td>3.0</td>
      <td>619.0</td>
      <td>14982.48</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00000550C665101A</td>
      <td>docked_bike</td>
      <td>2020-06-06 15:20:01</td>
      <td>2020-06-06 16:28:09</td>
      <td>Sheffield Ave &amp; Waveland Ave</td>
      <td>114</td>
      <td>Kedzie Ave &amp; Milwaukee Ave</td>
      <td>260</td>
      <td>41.949399</td>
      <td>-87.654529</td>
      <td>41.929567</td>
      <td>-87.707857</td>
      <td>casual</td>
      <td>7.0</td>
      <td>10808.0</td>
      <td>56896.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0000127970C84F62</td>
      <td>docked_bike</td>
      <td>2020-05-30 06:36:36</td>
      <td>2020-05-30 06:55:28</td>
      <td>Green St &amp; Madison St</td>
      <td>198</td>
      <td>Wells St &amp; Concord Ln</td>
      <td>289</td>
      <td>41.881900</td>
      <td>-87.648800</td>
      <td>41.912100</td>
      <td>-87.634700</td>
      <td>member</td>
      <td>7.0</td>
      <td>1892.0</td>
      <td>33329.42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00001A81D056B01B</td>
      <td>classic_bike</td>
      <td>2021-04-14 08:10:11</td>
      <td>2021-04-14 08:19:14</td>
      <td>Wood St &amp; Hubbard St</td>
      <td>13432</td>
      <td>Morgan St &amp; Lake St</td>
      <td>TA1306000015</td>
      <td>41.889899</td>
      <td>-87.671473</td>
      <td>41.885483</td>
      <td>-87.652305</td>
      <td>member</td>
      <td>4.0</td>
      <td>903.0</td>
      <td>19670.18</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df['duration'] <= 0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ride_id</th>
      <th>rideable_type</th>
      <th>started_at</th>
      <th>ended_at</th>
      <th>start_station_name</th>
      <th>start_station_id</th>
      <th>end_station_name</th>
      <th>end_station_id</th>
      <th>start_lat</th>
      <th>start_lng</th>
      <th>end_lat</th>
      <th>end_lng</th>
      <th>member_casual</th>
      <th>day_week_start</th>
      <th>duration</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>311</th>
      <td>0003D0AA1F7CEAC8</td>
      <td>docked_bike</td>
      <td>2020-08-23 12:10:56</td>
      <td>2020-08-23 12:10:40</td>
      <td>Clark St &amp; Chicago Ave</td>
      <td>337</td>
      <td>Noble St &amp; Milwaukee Ave</td>
      <td>29</td>
      <td>41.896750</td>
      <td>-87.630890</td>
      <td>41.900680</td>
      <td>-87.662600</td>
      <td>casual</td>
      <td>1.0</td>
      <td>-16.0</td>
      <td>31952.61</td>
    </tr>
    <tr>
      <th>2027</th>
      <td>00183B358D2A454A</td>
      <td>docked_bike</td>
      <td>2020-06-07 14:28:51</td>
      <td>2020-06-07 14:28:44</td>
      <td>Clark St &amp; Wellington Ave</td>
      <td>156</td>
      <td>Milwaukee Ave &amp; Grand Ave</td>
      <td>84</td>
      <td>41.936496</td>
      <td>-87.647538</td>
      <td>41.891578</td>
      <td>-87.648384</td>
      <td>casual</td>
      <td>1.0</td>
      <td>-7.0</td>
      <td>44925.97</td>
    </tr>
    <tr>
      <th>4872</th>
      <td>003A706B9EDAF788</td>
      <td>docked_bike</td>
      <td>2020-07-16 19:21:11</td>
      <td>2020-07-16 19:21:02</td>
      <td>Ritchie Ct &amp; Banks St</td>
      <td>180</td>
      <td>LaSalle St &amp; Illinois St</td>
      <td>181</td>
      <td>41.906866</td>
      <td>-87.626217</td>
      <td>41.890762</td>
      <td>-87.631697</td>
      <td>member</td>
      <td>5.0</td>
      <td>-9.0</td>
      <td>17010.86</td>
    </tr>
    <tr>
      <th>5071</th>
      <td>003CB4B59125E2F8</td>
      <td>docked_bike</td>
      <td>2020-08-22 14:35:01</td>
      <td>2020-08-22 14:34:50</td>
      <td>Wilton Ave &amp; Diversey Pkwy</td>
      <td>13</td>
      <td>Lake Shore Dr &amp; Belmont Ave</td>
      <td>334</td>
      <td>41.932418</td>
      <td>-87.652705</td>
      <td>41.940775</td>
      <td>-87.639192</td>
      <td>member</td>
      <td>7.0</td>
      <td>-51.0</td>
      <td>15888.38</td>
    </tr>
    <tr>
      <th>5640</th>
      <td>0043859A4D6DC837</td>
      <td>docked_bike</td>
      <td>2020-06-23 18:12:25</td>
      <td>2020-06-23 18:12:25</td>
      <td>Burnham Harbor</td>
      <td>4</td>
      <td>Field Blvd &amp; South Water St</td>
      <td>7</td>
      <td>41.856268</td>
      <td>-87.613348</td>
      <td>41.886349</td>
      <td>-87.617517</td>
      <td>member</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>30368.46</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5513940</th>
      <td>FFF3424C0EF50E94</td>
      <td>docked_bike</td>
      <td>2020-10-09 17:08:53</td>
      <td>2020-10-09 17:06:14</td>
      <td>Green St &amp; Madison St</td>
      <td>198</td>
      <td>Kingsbury St &amp; Kinzie St</td>
      <td>133</td>
      <td>41.881892</td>
      <td>-87.648789</td>
      <td>41.889177</td>
      <td>-87.638506</td>
      <td>member</td>
      <td>6.0</td>
      <td>-239.0</td>
      <td>12602.11</td>
    </tr>
    <tr>
      <th>5514326</th>
      <td>FFF73C3D40D75146</td>
      <td>docked_bike</td>
      <td>2020-10-02 08:02:59</td>
      <td>2020-10-02 08:02:04</td>
      <td>Southport Ave &amp; Wrightwood Ave</td>
      <td>190</td>
      <td>Southport Ave &amp; Belmont Ave</td>
      <td>154</td>
      <td>41.928773</td>
      <td>-87.663913</td>
      <td>41.938943</td>
      <td>-87.663866</td>
      <td>casual</td>
      <td>6.0</td>
      <td>-55.0</td>
      <td>10170.11</td>
    </tr>
    <tr>
      <th>5514414</th>
      <td>FFF850D8D4C3EFC9</td>
      <td>electric_bike</td>
      <td>2020-10-16 16:45:00</td>
      <td>2020-10-16 15:58:41</td>
      <td>Clifton Ave &amp; Armitage Ave</td>
      <td>223</td>
      <td>Larrabee St &amp; Webster Ave</td>
      <td>144</td>
      <td>41.918163</td>
      <td>-87.657064</td>
      <td>41.921848</td>
      <td>-87.644057</td>
      <td>casual</td>
      <td>6.0</td>
      <td>-8659.0</td>
      <td>13518.55</td>
    </tr>
    <tr>
      <th>5514544</th>
      <td>FFF9CB518645F387</td>
      <td>docked_bike</td>
      <td>2020-11-09 18:54:16</td>
      <td>2020-11-09 18:53:48</td>
      <td>Wabash Ave &amp; 9th St</td>
      <td>321</td>
      <td>Indiana Ave &amp; 26th St</td>
      <td>147</td>
      <td>41.870769</td>
      <td>-87.625734</td>
      <td>41.845687</td>
      <td>-87.622481</td>
      <td>member</td>
      <td>2.0</td>
      <td>-68.0</td>
      <td>25292.07</td>
    </tr>
    <tr>
      <th>5515018</th>
      <td>FFFF33C12C91FAC9</td>
      <td>docked_bike</td>
      <td>2020-03-12 09:04:58</td>
      <td>2020-03-12 09:04:45</td>
      <td>HQ QR</td>
      <td>675</td>
      <td>HQ QR</td>
      <td>675</td>
      <td>41.889900</td>
      <td>-87.680300</td>
      <td>41.889900</td>
      <td>-87.680300</td>
      <td>casual</td>
      <td>5.0</td>
      <td>-13.0</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
<p>11306 rows × 16 columns</p>
</div>




```python
df.drop(df.index[df['duration'] <= 0],inplace =True)
```


```python
df[df['duration'] <= 0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ride_id</th>
      <th>rideable_type</th>
      <th>started_at</th>
      <th>ended_at</th>
      <th>start_station_name</th>
      <th>start_station_id</th>
      <th>end_station_name</th>
      <th>end_station_id</th>
      <th>start_lat</th>
      <th>start_lng</th>
      <th>end_lat</th>
      <th>end_lng</th>
      <th>member_casual</th>
      <th>day_week_start</th>
      <th>duration</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ride_id</th>
      <th>rideable_type</th>
      <th>started_at</th>
      <th>ended_at</th>
      <th>start_station_name</th>
      <th>start_station_id</th>
      <th>end_station_name</th>
      <th>end_station_id</th>
      <th>start_lat</th>
      <th>start_lng</th>
      <th>end_lat</th>
      <th>end_lng</th>
      <th>member_casual</th>
      <th>day_week_start</th>
      <th>duration</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000001004784CD35</td>
      <td>docked_bike</td>
      <td>2020-07-22 15:38:23</td>
      <td>2020-07-22 15:56:47</td>
      <td>Wolcott (Ravenswood) Ave &amp; Montrose Ave</td>
      <td>238</td>
      <td>Southport Ave &amp; Clybourn Ave</td>
      <td>307</td>
      <td>41.961406</td>
      <td>-87.676169</td>
      <td>41.920771</td>
      <td>-87.663712</td>
      <td>casual</td>
      <td>4.0</td>
      <td>1824.0</td>
      <td>42501.53</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000002EBE159AE82</td>
      <td>electric_bike</td>
      <td>2021-06-22 17:25:15</td>
      <td>2021-06-22 17:31:34</td>
      <td>Clinton St &amp; Jackson Blvd</td>
      <td>638</td>
      <td>Milwaukee Ave &amp; Grand Ave</td>
      <td>13033</td>
      <td>41.878261</td>
      <td>-87.641155</td>
      <td>41.891495</td>
      <td>-87.648179</td>
      <td>member</td>
      <td>3.0</td>
      <td>619.0</td>
      <td>14982.48</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00000550C665101A</td>
      <td>docked_bike</td>
      <td>2020-06-06 15:20:01</td>
      <td>2020-06-06 16:28:09</td>
      <td>Sheffield Ave &amp; Waveland Ave</td>
      <td>114</td>
      <td>Kedzie Ave &amp; Milwaukee Ave</td>
      <td>260</td>
      <td>41.949399</td>
      <td>-87.654529</td>
      <td>41.929567</td>
      <td>-87.707857</td>
      <td>casual</td>
      <td>7.0</td>
      <td>10808.0</td>
      <td>56896.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0000127970C84F62</td>
      <td>docked_bike</td>
      <td>2020-05-30 06:36:36</td>
      <td>2020-05-30 06:55:28</td>
      <td>Green St &amp; Madison St</td>
      <td>198</td>
      <td>Wells St &amp; Concord Ln</td>
      <td>289</td>
      <td>41.881900</td>
      <td>-87.648800</td>
      <td>41.912100</td>
      <td>-87.634700</td>
      <td>member</td>
      <td>7.0</td>
      <td>1892.0</td>
      <td>33329.42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00001A81D056B01B</td>
      <td>classic_bike</td>
      <td>2021-04-14 08:10:11</td>
      <td>2021-04-14 08:19:14</td>
      <td>Wood St &amp; Hubbard St</td>
      <td>13432</td>
      <td>Morgan St &amp; Lake St</td>
      <td>TA1306000015</td>
      <td>41.889899</td>
      <td>-87.671473</td>
      <td>41.885483</td>
      <td>-87.652305</td>
      <td>member</td>
      <td>4.0</td>
      <td>903.0</td>
      <td>19670.18</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df['distance']<0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ride_id</th>
      <th>rideable_type</th>
      <th>started_at</th>
      <th>ended_at</th>
      <th>start_station_name</th>
      <th>start_station_id</th>
      <th>end_station_name</th>
      <th>end_station_id</th>
      <th>start_lat</th>
      <th>start_lng</th>
      <th>end_lat</th>
      <th>end_lng</th>
      <th>member_casual</th>
      <th>day_week_start</th>
      <th>duration</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>299</th>
      <td>0003A9FB01F7FD45</td>
      <td>docked_bike</td>
      <td>2020-08-16 13:25:56</td>
      <td>2020-08-16 13:33:28</td>
      <td>Lake Shore Dr &amp; Wellington Ave</td>
      <td>157</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.936688</td>
      <td>-87.636829</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>member</td>
      <td>1.0</td>
      <td>772.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>963</th>
      <td>000BB7D9D7A07F05</td>
      <td>classic_bike</td>
      <td>2021-05-23 02:10:46</td>
      <td>2021-05-23 02:10:59</td>
      <td>Halsted St &amp; Dickens Ave</td>
      <td>13192</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.919936</td>
      <td>-87.648830</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>casual</td>
      <td>1.0</td>
      <td>13.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>1167</th>
      <td>000E0BF00DF3C020</td>
      <td>classic_bike</td>
      <td>2021-04-09 18:17:43</td>
      <td>2021-04-10 19:17:37</td>
      <td>Latrobe Ave &amp; Chicago Ave</td>
      <td>642</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.894745</td>
      <td>-87.756895</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>casual</td>
      <td>6.0</td>
      <td>1009994.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>1453</th>
      <td>0011A31EF3CA864C_12</td>
      <td>docked_bike</td>
      <td>2020-12-02 15:05:38</td>
      <td>2020-12-03 16:05:29</td>
      <td>Sheridan Rd &amp; Columbia Ave</td>
      <td>660</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>42.004583</td>
      <td>-87.661406</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>member</td>
      <td>4.0</td>
      <td>1009991.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>1510</th>
      <td>00126450B12A886B</td>
      <td>classic_bike</td>
      <td>2021-05-21 18:07:54</td>
      <td>2021-05-22 19:07:44</td>
      <td>Green St &amp; Madison St</td>
      <td>TA1307000120</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.881892</td>
      <td>-87.648789</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>member</td>
      <td>6.0</td>
      <td>1009990.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5511589</th>
      <td>FFD707721D7C0824</td>
      <td>docked_bike</td>
      <td>2020-07-20 19:46:23</td>
      <td>2020-07-20 20:09:01</td>
      <td>Theater on the Lake</td>
      <td>177</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.926277</td>
      <td>-87.630834</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>casual</td>
      <td>2.0</td>
      <td>6278.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>5511725</th>
      <td>FFD892DDA304E510</td>
      <td>classic_bike</td>
      <td>2021-03-09 02:27:36</td>
      <td>2021-03-10 03:27:28</td>
      <td>Rush St &amp; Superior St</td>
      <td>15530</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.895765</td>
      <td>-87.625908</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>casual</td>
      <td>3.0</td>
      <td>1009992.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>5513820</th>
      <td>FFF1DAE49435E6EE</td>
      <td>docked_bike</td>
      <td>2020-10-30 04:47:38</td>
      <td>2020-10-30 11:02:12</td>
      <td>Washtenaw Ave &amp; Ogden Ave</td>
      <td>437</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.861930</td>
      <td>-87.693450</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>member</td>
      <td>6.0</td>
      <td>65474.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>5514100</th>
      <td>FFF4D9E9B8712894</td>
      <td>docked_bike</td>
      <td>2020-11-25 07:41:56</td>
      <td>2020-11-25 08:29:43</td>
      <td>Wolcott (Ravenswood) Ave &amp; Montrose Ave</td>
      <td>238</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.961406</td>
      <td>-87.676169</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>member</td>
      <td>4.0</td>
      <td>8787.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>5514357</th>
      <td>FFF7A4A2A4F4FA32</td>
      <td>docked_bike</td>
      <td>2020-11-03 10:05:46</td>
      <td>2020-11-03 10:21:16</td>
      <td>Clark St &amp; Armitage Ave</td>
      <td>94</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.918306</td>
      <td>-87.636282</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>member</td>
      <td>3.0</td>
      <td>1570.0</td>
      <td>-1.0</td>
    </tr>
  </tbody>
</table>
<p>6113 rows × 16 columns</p>
</div>




```python
df.drop(df.index[df['distance'] <= 0],inplace =True)
```


```python
df[df['distance']<0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ride_id</th>
      <th>rideable_type</th>
      <th>started_at</th>
      <th>ended_at</th>
      <th>start_station_name</th>
      <th>start_station_id</th>
      <th>end_station_name</th>
      <th>end_station_id</th>
      <th>start_lat</th>
      <th>start_lng</th>
      <th>end_lat</th>
      <th>end_lng</th>
      <th>member_casual</th>
      <th>day_week_start</th>
      <th>duration</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ride_id</th>
      <th>rideable_type</th>
      <th>started_at</th>
      <th>ended_at</th>
      <th>start_station_name</th>
      <th>start_station_id</th>
      <th>end_station_name</th>
      <th>end_station_id</th>
      <th>start_lat</th>
      <th>start_lng</th>
      <th>end_lat</th>
      <th>end_lng</th>
      <th>member_casual</th>
      <th>day_week_start</th>
      <th>duration</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000001004784CD35</td>
      <td>docked_bike</td>
      <td>2020-07-22 15:38:23</td>
      <td>2020-07-22 15:56:47</td>
      <td>Wolcott (Ravenswood) Ave &amp; Montrose Ave</td>
      <td>238</td>
      <td>Southport Ave &amp; Clybourn Ave</td>
      <td>307</td>
      <td>41.961406</td>
      <td>-87.676169</td>
      <td>41.920771</td>
      <td>-87.663712</td>
      <td>casual</td>
      <td>4.0</td>
      <td>1824.0</td>
      <td>42501.53</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000002EBE159AE82</td>
      <td>electric_bike</td>
      <td>2021-06-22 17:25:15</td>
      <td>2021-06-22 17:31:34</td>
      <td>Clinton St &amp; Jackson Blvd</td>
      <td>638</td>
      <td>Milwaukee Ave &amp; Grand Ave</td>
      <td>13033</td>
      <td>41.878261</td>
      <td>-87.641155</td>
      <td>41.891495</td>
      <td>-87.648179</td>
      <td>member</td>
      <td>3.0</td>
      <td>619.0</td>
      <td>14982.48</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00000550C665101A</td>
      <td>docked_bike</td>
      <td>2020-06-06 15:20:01</td>
      <td>2020-06-06 16:28:09</td>
      <td>Sheffield Ave &amp; Waveland Ave</td>
      <td>114</td>
      <td>Kedzie Ave &amp; Milwaukee Ave</td>
      <td>260</td>
      <td>41.949399</td>
      <td>-87.654529</td>
      <td>41.929567</td>
      <td>-87.707857</td>
      <td>casual</td>
      <td>7.0</td>
      <td>10808.0</td>
      <td>56896.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0000127970C84F62</td>
      <td>docked_bike</td>
      <td>2020-05-30 06:36:36</td>
      <td>2020-05-30 06:55:28</td>
      <td>Green St &amp; Madison St</td>
      <td>198</td>
      <td>Wells St &amp; Concord Ln</td>
      <td>289</td>
      <td>41.881900</td>
      <td>-87.648800</td>
      <td>41.912100</td>
      <td>-87.634700</td>
      <td>member</td>
      <td>7.0</td>
      <td>1892.0</td>
      <td>33329.42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00001A81D056B01B</td>
      <td>classic_bike</td>
      <td>2021-04-14 08:10:11</td>
      <td>2021-04-14 08:19:14</td>
      <td>Wood St &amp; Hubbard St</td>
      <td>13432</td>
      <td>Morgan St &amp; Lake St</td>
      <td>TA1306000015</td>
      <td>41.889899</td>
      <td>-87.671473</td>
      <td>41.885483</td>
      <td>-87.652305</td>
      <td>member</td>
      <td>4.0</td>
      <td>903.0</td>
      <td>19670.18</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.isnull().sum()
```




    ride_id                    0
    rideable_type              1
    started_at                 1
    ended_at                   1
    start_station_name    241975
    start_station_id      242532
    end_station_name      269491
    end_station_id        269890
    start_lat                  1
    start_lng                  1
    end_lat                    1
    end_lng                    1
    member_casual              1
    day_week_start             1
    duration                   1
    distance                   1
    dtype: int64




```python
df = df.dropna()
```


```python
 df.isnull().sum()
```




    ride_id               0
    rideable_type         0
    started_at            0
    ended_at              0
    start_station_name    0
    start_station_id      0
    end_station_name      0
    end_station_id        0
    start_lat             0
    start_lng             0
    end_lat               0
    end_lng               0
    member_casual         0
    day_week_start        0
    duration              0
    distance              0
    dtype: int64




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ride_id</th>
      <th>rideable_type</th>
      <th>started_at</th>
      <th>ended_at</th>
      <th>start_station_name</th>
      <th>start_station_id</th>
      <th>end_station_name</th>
      <th>end_station_id</th>
      <th>start_lat</th>
      <th>start_lng</th>
      <th>end_lat</th>
      <th>end_lng</th>
      <th>member_casual</th>
      <th>day_week_start</th>
      <th>duration</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000001004784CD35</td>
      <td>docked_bike</td>
      <td>2020-07-22 15:38:23</td>
      <td>2020-07-22 15:56:47</td>
      <td>Wolcott (Ravenswood) Ave &amp; Montrose Ave</td>
      <td>238</td>
      <td>Southport Ave &amp; Clybourn Ave</td>
      <td>307</td>
      <td>41.961406</td>
      <td>-87.676169</td>
      <td>41.920771</td>
      <td>-87.663712</td>
      <td>casual</td>
      <td>4.0</td>
      <td>1824.0</td>
      <td>42501.53</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000002EBE159AE82</td>
      <td>electric_bike</td>
      <td>2021-06-22 17:25:15</td>
      <td>2021-06-22 17:31:34</td>
      <td>Clinton St &amp; Jackson Blvd</td>
      <td>638</td>
      <td>Milwaukee Ave &amp; Grand Ave</td>
      <td>13033</td>
      <td>41.878261</td>
      <td>-87.641155</td>
      <td>41.891495</td>
      <td>-87.648179</td>
      <td>member</td>
      <td>3.0</td>
      <td>619.0</td>
      <td>14982.48</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00000550C665101A</td>
      <td>docked_bike</td>
      <td>2020-06-06 15:20:01</td>
      <td>2020-06-06 16:28:09</td>
      <td>Sheffield Ave &amp; Waveland Ave</td>
      <td>114</td>
      <td>Kedzie Ave &amp; Milwaukee Ave</td>
      <td>260</td>
      <td>41.949399</td>
      <td>-87.654529</td>
      <td>41.929567</td>
      <td>-87.707857</td>
      <td>casual</td>
      <td>7.0</td>
      <td>10808.0</td>
      <td>56896.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0000127970C84F62</td>
      <td>docked_bike</td>
      <td>2020-05-30 06:36:36</td>
      <td>2020-05-30 06:55:28</td>
      <td>Green St &amp; Madison St</td>
      <td>198</td>
      <td>Wells St &amp; Concord Ln</td>
      <td>289</td>
      <td>41.881900</td>
      <td>-87.648800</td>
      <td>41.912100</td>
      <td>-87.634700</td>
      <td>member</td>
      <td>7.0</td>
      <td>1892.0</td>
      <td>33329.42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00001A81D056B01B</td>
      <td>classic_bike</td>
      <td>2021-04-14 08:10:11</td>
      <td>2021-04-14 08:19:14</td>
      <td>Wood St &amp; Hubbard St</td>
      <td>13432</td>
      <td>Morgan St &amp; Lake St</td>
      <td>TA1306000015</td>
      <td>41.889899</td>
      <td>-87.671473</td>
      <td>41.885483</td>
      <td>-87.652305</td>
      <td>member</td>
      <td>4.0</td>
      <td>903.0</td>
      <td>19670.18</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df.duplicated()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ride_id</th>
      <th>rideable_type</th>
      <th>started_at</th>
      <th>ended_at</th>
      <th>start_station_name</th>
      <th>start_station_id</th>
      <th>end_station_name</th>
      <th>end_station_id</th>
      <th>start_lat</th>
      <th>start_lng</th>
      <th>end_lat</th>
      <th>end_lng</th>
      <th>member_casual</th>
      <th>day_week_start</th>
      <th>duration</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
df['day_week_start'].replace({1.0:'sunday',2.0:'monday',3.0:'tuesday',4.0:'wednesday',5.0:'thursday',6.0:'friday',7.0:'saturday'},inplace = True)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ride_id</th>
      <th>rideable_type</th>
      <th>started_at</th>
      <th>ended_at</th>
      <th>start_station_name</th>
      <th>start_station_id</th>
      <th>end_station_name</th>
      <th>end_station_id</th>
      <th>start_lat</th>
      <th>start_lng</th>
      <th>end_lat</th>
      <th>end_lng</th>
      <th>member_casual</th>
      <th>day_week_start</th>
      <th>duration</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000001004784CD35</td>
      <td>docked_bike</td>
      <td>2020-07-22 15:38:23</td>
      <td>2020-07-22 15:56:47</td>
      <td>Wolcott (Ravenswood) Ave &amp; Montrose Ave</td>
      <td>238</td>
      <td>Southport Ave &amp; Clybourn Ave</td>
      <td>307</td>
      <td>41.961406</td>
      <td>-87.676169</td>
      <td>41.920771</td>
      <td>-87.663712</td>
      <td>casual</td>
      <td>wednesday</td>
      <td>1824.0</td>
      <td>42501.53</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000002EBE159AE82</td>
      <td>electric_bike</td>
      <td>2021-06-22 17:25:15</td>
      <td>2021-06-22 17:31:34</td>
      <td>Clinton St &amp; Jackson Blvd</td>
      <td>638</td>
      <td>Milwaukee Ave &amp; Grand Ave</td>
      <td>13033</td>
      <td>41.878261</td>
      <td>-87.641155</td>
      <td>41.891495</td>
      <td>-87.648179</td>
      <td>member</td>
      <td>tuesday</td>
      <td>619.0</td>
      <td>14982.48</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00000550C665101A</td>
      <td>docked_bike</td>
      <td>2020-06-06 15:20:01</td>
      <td>2020-06-06 16:28:09</td>
      <td>Sheffield Ave &amp; Waveland Ave</td>
      <td>114</td>
      <td>Kedzie Ave &amp; Milwaukee Ave</td>
      <td>260</td>
      <td>41.949399</td>
      <td>-87.654529</td>
      <td>41.929567</td>
      <td>-87.707857</td>
      <td>casual</td>
      <td>saturday</td>
      <td>10808.0</td>
      <td>56896.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0000127970C84F62</td>
      <td>docked_bike</td>
      <td>2020-05-30 06:36:36</td>
      <td>2020-05-30 06:55:28</td>
      <td>Green St &amp; Madison St</td>
      <td>198</td>
      <td>Wells St &amp; Concord Ln</td>
      <td>289</td>
      <td>41.881900</td>
      <td>-87.648800</td>
      <td>41.912100</td>
      <td>-87.634700</td>
      <td>member</td>
      <td>saturday</td>
      <td>1892.0</td>
      <td>33329.42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00001A81D056B01B</td>
      <td>classic_bike</td>
      <td>2021-04-14 08:10:11</td>
      <td>2021-04-14 08:19:14</td>
      <td>Wood St &amp; Hubbard St</td>
      <td>13432</td>
      <td>Morgan St &amp; Lake St</td>
      <td>TA1306000015</td>
      <td>41.889899</td>
      <td>-87.671473</td>
      <td>41.885483</td>
      <td>-87.652305</td>
      <td>member</td>
      <td>wednesday</td>
      <td>903.0</td>
      <td>19670.18</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df.drop(['start_lat','start_lng','end_lat','end_lng'],axis = 1)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ride_id</th>
      <th>rideable_type</th>
      <th>started_at</th>
      <th>ended_at</th>
      <th>start_station_name</th>
      <th>start_station_id</th>
      <th>end_station_name</th>
      <th>end_station_id</th>
      <th>member_casual</th>
      <th>day_week_start</th>
      <th>duration</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000001004784CD35</td>
      <td>docked_bike</td>
      <td>2020-07-22 15:38:23</td>
      <td>2020-07-22 15:56:47</td>
      <td>Wolcott (Ravenswood) Ave &amp; Montrose Ave</td>
      <td>238</td>
      <td>Southport Ave &amp; Clybourn Ave</td>
      <td>307</td>
      <td>casual</td>
      <td>wednesday</td>
      <td>1824.0</td>
      <td>42501.53</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000002EBE159AE82</td>
      <td>electric_bike</td>
      <td>2021-06-22 17:25:15</td>
      <td>2021-06-22 17:31:34</td>
      <td>Clinton St &amp; Jackson Blvd</td>
      <td>638</td>
      <td>Milwaukee Ave &amp; Grand Ave</td>
      <td>13033</td>
      <td>member</td>
      <td>tuesday</td>
      <td>619.0</td>
      <td>14982.48</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00000550C665101A</td>
      <td>docked_bike</td>
      <td>2020-06-06 15:20:01</td>
      <td>2020-06-06 16:28:09</td>
      <td>Sheffield Ave &amp; Waveland Ave</td>
      <td>114</td>
      <td>Kedzie Ave &amp; Milwaukee Ave</td>
      <td>260</td>
      <td>casual</td>
      <td>saturday</td>
      <td>10808.0</td>
      <td>56896.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0000127970C84F62</td>
      <td>docked_bike</td>
      <td>2020-05-30 06:36:36</td>
      <td>2020-05-30 06:55:28</td>
      <td>Green St &amp; Madison St</td>
      <td>198</td>
      <td>Wells St &amp; Concord Ln</td>
      <td>289</td>
      <td>member</td>
      <td>saturday</td>
      <td>1892.0</td>
      <td>33329.42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00001A81D056B01B</td>
      <td>classic_bike</td>
      <td>2021-04-14 08:10:11</td>
      <td>2021-04-14 08:19:14</td>
      <td>Wood St &amp; Hubbard St</td>
      <td>13432</td>
      <td>Morgan St &amp; Lake St</td>
      <td>TA1306000015</td>
      <td>member</td>
      <td>wednesday</td>
      <td>903.0</td>
      <td>19670.18</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_member = df
df_causal = df
df_both = df
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ride_id</th>
      <th>rideable_type</th>
      <th>started_at</th>
      <th>ended_at</th>
      <th>start_station_name</th>
      <th>start_station_id</th>
      <th>end_station_name</th>
      <th>end_station_id</th>
      <th>member_casual</th>
      <th>day_week_start</th>
      <th>duration</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000001004784CD35</td>
      <td>docked_bike</td>
      <td>2020-07-22 15:38:23</td>
      <td>2020-07-22 15:56:47</td>
      <td>Wolcott (Ravenswood) Ave &amp; Montrose Ave</td>
      <td>238</td>
      <td>Southport Ave &amp; Clybourn Ave</td>
      <td>307</td>
      <td>casual</td>
      <td>wednesday</td>
      <td>1824.0</td>
      <td>42501.53</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000002EBE159AE82</td>
      <td>electric_bike</td>
      <td>2021-06-22 17:25:15</td>
      <td>2021-06-22 17:31:34</td>
      <td>Clinton St &amp; Jackson Blvd</td>
      <td>638</td>
      <td>Milwaukee Ave &amp; Grand Ave</td>
      <td>13033</td>
      <td>member</td>
      <td>tuesday</td>
      <td>619.0</td>
      <td>14982.48</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00000550C665101A</td>
      <td>docked_bike</td>
      <td>2020-06-06 15:20:01</td>
      <td>2020-06-06 16:28:09</td>
      <td>Sheffield Ave &amp; Waveland Ave</td>
      <td>114</td>
      <td>Kedzie Ave &amp; Milwaukee Ave</td>
      <td>260</td>
      <td>casual</td>
      <td>saturday</td>
      <td>10808.0</td>
      <td>56896.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0000127970C84F62</td>
      <td>docked_bike</td>
      <td>2020-05-30 06:36:36</td>
      <td>2020-05-30 06:55:28</td>
      <td>Green St &amp; Madison St</td>
      <td>198</td>
      <td>Wells St &amp; Concord Ln</td>
      <td>289</td>
      <td>member</td>
      <td>saturday</td>
      <td>1892.0</td>
      <td>33329.42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00001A81D056B01B</td>
      <td>classic_bike</td>
      <td>2021-04-14 08:10:11</td>
      <td>2021-04-14 08:19:14</td>
      <td>Wood St &amp; Hubbard St</td>
      <td>13432</td>
      <td>Morgan St &amp; Lake St</td>
      <td>TA1306000015</td>
      <td>member</td>
      <td>wednesday</td>
      <td>903.0</td>
      <td>19670.18</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>duration</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.685565e+06</td>
      <td>4.685565e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.644010e+04</td>
      <td>2.369751e+04</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.538609e+07</td>
      <td>1.876522e+04</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000e+00</td>
      <td>1.000000e-02</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8.020000e+02</td>
      <td>1.074132e+04</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.586000e+03</td>
      <td>1.844213e+04</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.121000e+03</td>
      <td>3.146654e+04</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.875942e+09</td>
      <td>4.494231e+05</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ride_id</th>
      <th>rideable_type</th>
      <th>started_at</th>
      <th>ended_at</th>
      <th>start_station_name</th>
      <th>start_station_id</th>
      <th>end_station_name</th>
      <th>end_station_id</th>
      <th>member_casual</th>
      <th>day_week_start</th>
      <th>duration</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000001004784CD35</td>
      <td>docked_bike</td>
      <td>2020-07-22 15:38:23</td>
      <td>2020-07-22 15:56:47</td>
      <td>Wolcott (Ravenswood) Ave &amp; Montrose Ave</td>
      <td>238</td>
      <td>Southport Ave &amp; Clybourn Ave</td>
      <td>307</td>
      <td>casual</td>
      <td>wednesday</td>
      <td>1824.0</td>
      <td>42501.53</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000002EBE159AE82</td>
      <td>electric_bike</td>
      <td>2021-06-22 17:25:15</td>
      <td>2021-06-22 17:31:34</td>
      <td>Clinton St &amp; Jackson Blvd</td>
      <td>638</td>
      <td>Milwaukee Ave &amp; Grand Ave</td>
      <td>13033</td>
      <td>member</td>
      <td>tuesday</td>
      <td>619.0</td>
      <td>14982.48</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00000550C665101A</td>
      <td>docked_bike</td>
      <td>2020-06-06 15:20:01</td>
      <td>2020-06-06 16:28:09</td>
      <td>Sheffield Ave &amp; Waveland Ave</td>
      <td>114</td>
      <td>Kedzie Ave &amp; Milwaukee Ave</td>
      <td>260</td>
      <td>casual</td>
      <td>saturday</td>
      <td>10808.0</td>
      <td>56896.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0000127970C84F62</td>
      <td>docked_bike</td>
      <td>2020-05-30 06:36:36</td>
      <td>2020-05-30 06:55:28</td>
      <td>Green St &amp; Madison St</td>
      <td>198</td>
      <td>Wells St &amp; Concord Ln</td>
      <td>289</td>
      <td>member</td>
      <td>saturday</td>
      <td>1892.0</td>
      <td>33329.42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00001A81D056B01B</td>
      <td>classic_bike</td>
      <td>2021-04-14 08:10:11</td>
      <td>2021-04-14 08:19:14</td>
      <td>Wood St &amp; Hubbard St</td>
      <td>13432</td>
      <td>Morgan St &amp; Lake St</td>
      <td>TA1306000015</td>
      <td>member</td>
      <td>wednesday</td>
      <td>903.0</td>
      <td>19670.18</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['member_casual'].value_counts()
```




    member    2917439
    casual    1768126
    Name: member_casual, dtype: int64




```python
#there are 2,917,439 members- 62.26%
#there are 1,768,126 casual riders-37.74%
```


```python
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ride_id</th>
      <th>rideable_type</th>
      <th>started_at</th>
      <th>ended_at</th>
      <th>start_station_name</th>
      <th>start_station_id</th>
      <th>end_station_name</th>
      <th>end_station_id</th>
      <th>member_casual</th>
      <th>day_week_start</th>
      <th>duration</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000001004784CD35</td>
      <td>docked_bike</td>
      <td>2020-07-22 15:38:23</td>
      <td>2020-07-22 15:56:47</td>
      <td>Wolcott (Ravenswood) Ave &amp; Montrose Ave</td>
      <td>238</td>
      <td>Southport Ave &amp; Clybourn Ave</td>
      <td>307</td>
      <td>casual</td>
      <td>wednesday</td>
      <td>1824.0</td>
      <td>42501.53</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000002EBE159AE82</td>
      <td>electric_bike</td>
      <td>2021-06-22 17:25:15</td>
      <td>2021-06-22 17:31:34</td>
      <td>Clinton St &amp; Jackson Blvd</td>
      <td>638</td>
      <td>Milwaukee Ave &amp; Grand Ave</td>
      <td>13033</td>
      <td>member</td>
      <td>tuesday</td>
      <td>619.0</td>
      <td>14982.48</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00000550C665101A</td>
      <td>docked_bike</td>
      <td>2020-06-06 15:20:01</td>
      <td>2020-06-06 16:28:09</td>
      <td>Sheffield Ave &amp; Waveland Ave</td>
      <td>114</td>
      <td>Kedzie Ave &amp; Milwaukee Ave</td>
      <td>260</td>
      <td>casual</td>
      <td>saturday</td>
      <td>10808.0</td>
      <td>56896.25</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['distance'].mean()
```




    23697.50758100466




```python
df['distance'].max()
```




    449423.09




```python
df['day_week_start'].mode()
```




    0    saturday
    Name: day_week_start, dtype: object




```python
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ride_id</th>
      <th>rideable_type</th>
      <th>started_at</th>
      <th>ended_at</th>
      <th>start_station_name</th>
      <th>start_station_id</th>
      <th>end_station_name</th>
      <th>end_station_id</th>
      <th>member_casual</th>
      <th>day_week_start</th>
      <th>duration</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000001004784CD35</td>
      <td>docked_bike</td>
      <td>2020-07-22 15:38:23</td>
      <td>2020-07-22 15:56:47</td>
      <td>Wolcott (Ravenswood) Ave &amp; Montrose Ave</td>
      <td>238</td>
      <td>Southport Ave &amp; Clybourn Ave</td>
      <td>307</td>
      <td>casual</td>
      <td>wednesday</td>
      <td>1824.0</td>
      <td>42501.53</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000002EBE159AE82</td>
      <td>electric_bike</td>
      <td>2021-06-22 17:25:15</td>
      <td>2021-06-22 17:31:34</td>
      <td>Clinton St &amp; Jackson Blvd</td>
      <td>638</td>
      <td>Milwaukee Ave &amp; Grand Ave</td>
      <td>13033</td>
      <td>member</td>
      <td>tuesday</td>
      <td>619.0</td>
      <td>14982.48</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['start_station_name'].value_counts()
```




    Streeter Dr & Grand Ave            53336
    Clark St & Elm St                  43893
    Lake Shore Dr & North Blvd         38252
    Lake Shore Dr & Monroe St          38184
    Wells St & Concord Ln              37816
                                       ...  
    Stewart Ave & 63rd St (*)              3
    HQ QR                                  1
    N Clark St & W Elm St                  1
    N Hampden Ct & W Diversey Ave          1
    Lyft Driver Center Private Rack        1
    Name: start_station_name, Length: 715, dtype: int64




```python
# From above we can conclude that :-
# 'Streeter Dr & Grand Ave'  is the most visited station,
# 53336 riders have started their rides from this station 
```


```python
df['end_station_name'].value_counts()
```




    Streeter Dr & Grand Ave          56545
    Clark St & Elm St                43956
    Lake Shore Dr & North Blvd       40798
    Theater on the Lake              39712
    Wells St & Concord Ln            38637
                                     ...  
    N Clark St & W Elm St                2
    N Damen Ave & W Wabansia St          2
    Avenue L & 114th St                  1
    HQ QR                                1
    N Hampden Ct & W Diversey Ave        1
    Name: end_station_name, Length: 716, dtype: int64




```python
# From above we can conclude that :-
# 'Streeter Dr & Grand Ave'  is the most visited station,
# 53336 riders have ended their rides from this station 
```


```python
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ride_id</th>
      <th>rideable_type</th>
      <th>started_at</th>
      <th>ended_at</th>
      <th>start_station_name</th>
      <th>start_station_id</th>
      <th>end_station_name</th>
      <th>end_station_id</th>
      <th>member_casual</th>
      <th>day_week_start</th>
      <th>duration</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000001004784CD35</td>
      <td>docked_bike</td>
      <td>2020-07-22 15:38:23</td>
      <td>2020-07-22 15:56:47</td>
      <td>Wolcott (Ravenswood) Ave &amp; Montrose Ave</td>
      <td>238</td>
      <td>Southport Ave &amp; Clybourn Ave</td>
      <td>307</td>
      <td>casual</td>
      <td>wednesday</td>
      <td>1824.0</td>
      <td>42501.53</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000002EBE159AE82</td>
      <td>electric_bike</td>
      <td>2021-06-22 17:25:15</td>
      <td>2021-06-22 17:31:34</td>
      <td>Clinton St &amp; Jackson Blvd</td>
      <td>638</td>
      <td>Milwaukee Ave &amp; Grand Ave</td>
      <td>13033</td>
      <td>member</td>
      <td>tuesday</td>
      <td>619.0</td>
      <td>14982.48</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['rideable_type'].value_counts()
```




    docked_bike      2776356
    classic_bike     1205865
    electric_bike     703344
    Name: rideable_type, dtype: int64




```python
# From above result we can conclude that :-
# 'docked_bike' is the most preferable rideable_type.
```


```python
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ride_id</th>
      <th>rideable_type</th>
      <th>started_at</th>
      <th>ended_at</th>
      <th>start_station_name</th>
      <th>start_station_id</th>
      <th>end_station_name</th>
      <th>end_station_id</th>
      <th>member_casual</th>
      <th>day_week_start</th>
      <th>duration</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000001004784CD35</td>
      <td>docked_bike</td>
      <td>2020-07-22 15:38:23</td>
      <td>2020-07-22 15:56:47</td>
      <td>Wolcott (Ravenswood) Ave &amp; Montrose Ave</td>
      <td>238</td>
      <td>Southport Ave &amp; Clybourn Ave</td>
      <td>307</td>
      <td>casual</td>
      <td>wednesday</td>
      <td>1824.0</td>
      <td>42501.53</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000002EBE159AE82</td>
      <td>electric_bike</td>
      <td>2021-06-22 17:25:15</td>
      <td>2021-06-22 17:31:34</td>
      <td>Clinton St &amp; Jackson Blvd</td>
      <td>638</td>
      <td>Milwaukee Ave &amp; Grand Ave</td>
      <td>13033</td>
      <td>member</td>
      <td>tuesday</td>
      <td>619.0</td>
      <td>14982.48</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['day_week_start'].value_counts()
```




    saturday     833136
    sunday       706254
    friday       686002
    wednesday    641247
    thursday     621602
    tuesday      615493
    monday       581831
    Name: day_week_start, dtype: int64




```python
# From above result we can conclude that :-
# 'saturday' is the busiest day of the week throught out the year.
```


```python
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ride_id</th>
      <th>rideable_type</th>
      <th>started_at</th>
      <th>ended_at</th>
      <th>start_station_name</th>
      <th>start_station_id</th>
      <th>end_station_name</th>
      <th>end_station_id</th>
      <th>member_casual</th>
      <th>day_week_start</th>
      <th>duration</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000001004784CD35</td>
      <td>docked_bike</td>
      <td>2020-07-22 15:38:23</td>
      <td>2020-07-22 15:56:47</td>
      <td>Wolcott (Ravenswood) Ave &amp; Montrose Ave</td>
      <td>238</td>
      <td>Southport Ave &amp; Clybourn Ave</td>
      <td>307</td>
      <td>casual</td>
      <td>wednesday</td>
      <td>1824.0</td>
      <td>42501.53</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000002EBE159AE82</td>
      <td>electric_bike</td>
      <td>2021-06-22 17:25:15</td>
      <td>2021-06-22 17:31:34</td>
      <td>Clinton St &amp; Jackson Blvd</td>
      <td>638</td>
      <td>Milwaukee Ave &amp; Grand Ave</td>
      <td>13033</td>
      <td>member</td>
      <td>tuesday</td>
      <td>619.0</td>
      <td>14982.48</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_member = df[df['member_casual'] == 'member']
```


```python
df_casual = df[df['member_casual'] == 'casual']
```


```python
df_member['distance'].mean()
```




    22985.715532444712




```python
df_member.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>duration</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.917439e+06</td>
      <td>2.917439e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.552066e+04</td>
      <td>2.298572e+04</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.163140e+07</td>
      <td>1.781781e+04</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000e+00</td>
      <td>1.000000e-02</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6.750000e+02</td>
      <td>1.037690e+04</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.224000e+03</td>
      <td>1.777553e+04</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.271000e+03</td>
      <td>3.051190e+04</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.869776e+09</td>
      <td>4.494231e+05</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_casual.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>duration</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.768126e+06</td>
      <td>1.768126e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.074577e+05</td>
      <td>2.487198e+04</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.010248e+07</td>
      <td>2.017689e+04</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000e+00</td>
      <td>1.000000e-02</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.196000e+03</td>
      <td>1.149538e+04</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.543000e+03</td>
      <td>1.971238e+04</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.790000e+03</td>
      <td>3.296709e+04</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.875942e+09</td>
      <td>3.042953e+05</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_member.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ride_id</th>
      <th>rideable_type</th>
      <th>started_at</th>
      <th>ended_at</th>
      <th>start_station_name</th>
      <th>start_station_id</th>
      <th>end_station_name</th>
      <th>end_station_id</th>
      <th>member_casual</th>
      <th>day_week_start</th>
      <th>duration</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>000002EBE159AE82</td>
      <td>electric_bike</td>
      <td>2021-06-22 17:25:15</td>
      <td>2021-06-22 17:31:34</td>
      <td>Clinton St &amp; Jackson Blvd</td>
      <td>638</td>
      <td>Milwaukee Ave &amp; Grand Ave</td>
      <td>13033</td>
      <td>member</td>
      <td>tuesday</td>
      <td>619.0</td>
      <td>14982.48</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0000127970C84F62</td>
      <td>docked_bike</td>
      <td>2020-05-30 06:36:36</td>
      <td>2020-05-30 06:55:28</td>
      <td>Green St &amp; Madison St</td>
      <td>198</td>
      <td>Wells St &amp; Concord Ln</td>
      <td>289</td>
      <td>member</td>
      <td>saturday</td>
      <td>1892.0</td>
      <td>33329.42</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_casual.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ride_id</th>
      <th>rideable_type</th>
      <th>started_at</th>
      <th>ended_at</th>
      <th>start_station_name</th>
      <th>start_station_id</th>
      <th>end_station_name</th>
      <th>end_station_id</th>
      <th>member_casual</th>
      <th>day_week_start</th>
      <th>duration</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000001004784CD35</td>
      <td>docked_bike</td>
      <td>2020-07-22 15:38:23</td>
      <td>2020-07-22 15:56:47</td>
      <td>Wolcott (Ravenswood) Ave &amp; Montrose Ave</td>
      <td>238</td>
      <td>Southport Ave &amp; Clybourn Ave</td>
      <td>307</td>
      <td>casual</td>
      <td>wednesday</td>
      <td>1824.0</td>
      <td>42501.53</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00000550C665101A</td>
      <td>docked_bike</td>
      <td>2020-06-06 15:20:01</td>
      <td>2020-06-06 16:28:09</td>
      <td>Sheffield Ave &amp; Waveland Ave</td>
      <td>114</td>
      <td>Kedzie Ave &amp; Milwaukee Ave</td>
      <td>260</td>
      <td>casual</td>
      <td>saturday</td>
      <td>10808.0</td>
      <td>56896.25</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_member['rideable_type'].value_counts()
#bikes riden by members only
#docked_bike was 1,727,733 members -59.22%
#classic_bike was 792,599 members - 27.17%
#electric_bike was 397,107 members - 13.61%
```




    docked_bike      1727733
    classic_bike      792599
    electric_bike     397107
    Name: rideable_type, dtype: int64




```python
df_casual['rideable_type'].value_counts()
#bikes riden by casual only
#docked_bike was 1,048,623 Casual -59.31%
#classic_bike was 413,266 Casual - 23.37%
#electric_bike was 306,237 Casual - 17.32%
```




    docked_bike      1048623
    classic_bike      413266
    electric_bike     306237
    Name: rideable_type, dtype: int64




```python
df_member['start_station_name'].value_counts()
```




    Clark St & Elm St                  28492
    Kingsbury St & Kinzie St           24650
    Wells St & Concord Ln              23030
    Dearborn St & Erie St              22423
    St. Clair St & Erie St             22228
                                       ...  
    Avenue O & 118th St                    1
    Vernon Ave & 107th St                  1
    Elizabeth St & 92nd St                 1
    Ashland Ave & 73rd St                  1
    Lyft Driver Center Private Rack        1
    Name: start_station_name, Length: 706, dtype: int64




```python
df_casual['start_station_name'].value_counts()
```




    Streeter Dr & Grand Ave       39555
    Millennium Park               26701
    Lake Shore Dr & Monroe St     25950
    Michigan Ave & Oak St         20887
    Lake Shore Dr & North Blvd    19380
                                  ...  
    Hampden Ct & Diversey Ave         3
    W 103rd St & S Avers Ave          3
    HQ QR                             1
    N Clark St & W Elm St             1
    Stewart Ave & 63rd St (*)         1
    Name: start_station_name, Length: 712, dtype: int64




```python
df_member['end_station_name'].value_counts()
```




    Clark St & Elm St           29235
    Kingsbury St & Kinzie St    24892
    St. Clair St & Erie St      23893
    Wells St & Concord Ln       23554
    Dearborn St & Erie St       23346
                                ...  
    Ashland Ave & 73rd St           1
    Yates Blvd & 93rd St            1
    N Clark St & W Elm St           1
    Avenue O & 134th St             1
    Loomis St & 89th St             1
    Name: end_station_name, Length: 709, dtype: int64




```python
df_casual['end_station_name'].value_counts()
```




    Streeter Dr & Grand Ave                             43370
    Millennium Park                                     28405
    Lake Shore Dr & Monroe St                           24336
    Lake Shore Dr & North Blvd                          22059
    Michigan Ave & Oak St                               22038
                                                        ...  
    Woodlawn & 103rd - Olive Harvey Vaccination Site        3
    Eggleston Ave & 69th St (*)                             2
    Avenue L & 114th St                                     1
    N Clark St & W Elm St                                   1
    N Hampden Ct & W Diversey Ave                           1
    Name: end_station_name, Length: 714, dtype: int64




```python
df_member['day_week_start'].value_counts()
```




    wednesday    447619
    tuesday      431542
    friday       430723
    thursday     426599
    saturday     419125
    monday       396422
    sunday       365409
    Name: day_week_start, dtype: int64




```python
df_casual['day_week_start'].value_counts()
```




    saturday     414011
    sunday       340845
    friday       255279
    thursday     195003
    wednesday    193628
    monday       185409
    tuesday      183951
    Name: day_week_start, dtype: int64




```python
import matplotlib.pyplot as plt
```


```python
from datetime import date
df['started_at'] = pd.to_datetime(df['started_at']).dt.normalize()

```


```python
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ride_id</th>
      <th>rideable_type</th>
      <th>started_at</th>
      <th>ended_at</th>
      <th>start_station_name</th>
      <th>start_station_id</th>
      <th>end_station_name</th>
      <th>end_station_id</th>
      <th>member_casual</th>
      <th>day_week_start</th>
      <th>duration</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000001004784CD35</td>
      <td>docked_bike</td>
      <td>2020-07-22</td>
      <td>2020-07-22 15:56:47</td>
      <td>Wolcott (Ravenswood) Ave &amp; Montrose Ave</td>
      <td>238</td>
      <td>Southport Ave &amp; Clybourn Ave</td>
      <td>307</td>
      <td>casual</td>
      <td>wednesday</td>
      <td>1824.0</td>
      <td>42501.53</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000002EBE159AE82</td>
      <td>electric_bike</td>
      <td>2021-06-22</td>
      <td>2021-06-22 17:31:34</td>
      <td>Clinton St &amp; Jackson Blvd</td>
      <td>638</td>
      <td>Milwaukee Ave &amp; Grand Ave</td>
      <td>13033</td>
      <td>member</td>
      <td>tuesday</td>
      <td>619.0</td>
      <td>14982.48</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00000550C665101A</td>
      <td>docked_bike</td>
      <td>2020-06-06</td>
      <td>2020-06-06 16:28:09</td>
      <td>Sheffield Ave &amp; Waveland Ave</td>
      <td>114</td>
      <td>Kedzie Ave &amp; Milwaukee Ave</td>
      <td>260</td>
      <td>casual</td>
      <td>saturday</td>
      <td>10808.0</td>
      <td>56896.25</td>
    </tr>
  </tbody>
</table>
</div>




```python
by_date = pd.Series(df['started_at']).value_counts().sort_index()
by_date.index = pd.DatetimeIndex(by_date.index)
df_date = by_date.rename_axis('date').reset_index(name='counts')
df_date
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-01-01</td>
      <td>1979</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-01-02</td>
      <td>6300</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-01-03</td>
      <td>5753</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01-04</td>
      <td>3083</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-01-05</td>
      <td>2879</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>540</th>
      <td>2021-06-26</td>
      <td>11138</td>
    </tr>
    <tr>
      <th>541</th>
      <td>2021-06-27</td>
      <td>22521</td>
    </tr>
    <tr>
      <th>542</th>
      <td>2021-06-28</td>
      <td>12717</td>
    </tr>
    <tr>
      <th>543</th>
      <td>2021-06-29</td>
      <td>13867</td>
    </tr>
    <tr>
      <th>544</th>
      <td>2021-06-30</td>
      <td>15008</td>
    </tr>
  </tbody>
</table>
<p>545 rows × 2 columns</p>
</div>




```python
fig= plt.figure()
ax = fig.add_axes([0,0,4,4])
ax.barh(df_date['date'],df_date['counts'])
plt.show()
```


    
![png](output_64_0.png)
    



```python
by_days = pd.Series(df['day_week_start']).value_counts().sort_index()
by_days.index = pd.Index(by_days.index)
dff_days = by_days.rename_axis('days').reset_index(name='counts')
dff_days
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>days</th>
      <th>counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>friday</td>
      <td>686002</td>
    </tr>
    <tr>
      <th>1</th>
      <td>monday</td>
      <td>581831</td>
    </tr>
    <tr>
      <th>2</th>
      <td>saturday</td>
      <td>833136</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sunday</td>
      <td>706254</td>
    </tr>
    <tr>
      <th>4</th>
      <td>thursday</td>
      <td>621602</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tuesday</td>
      <td>615493</td>
    </tr>
    <tr>
      <th>6</th>
      <td>wednesday</td>
      <td>641247</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.pie(dff_days['counts'],labels=dff_days['days'],autopct='%1.1f%%')
plt.title('Members and Casual')
plt.show()
```


    
![png](output_66_0.png)
    



```python
by_days = pd.Series(df_member['day_week_start']).value_counts().sort_index()
by_days.index = pd.Index(by_days.index)
df_member_days = by_days.rename_axis('days').reset_index(name='counts')
df_member_days
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>days</th>
      <th>counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>friday</td>
      <td>430723</td>
    </tr>
    <tr>
      <th>1</th>
      <td>monday</td>
      <td>396422</td>
    </tr>
    <tr>
      <th>2</th>
      <td>saturday</td>
      <td>419125</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sunday</td>
      <td>365409</td>
    </tr>
    <tr>
      <th>4</th>
      <td>thursday</td>
      <td>426599</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tuesday</td>
      <td>431542</td>
    </tr>
    <tr>
      <th>6</th>
      <td>wednesday</td>
      <td>447619</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.pie(df_member_days['counts'],labels=df_member_days['days'],autopct='%1.1f%%')
plt.title('Members')
plt.show()
```


    
![png](output_68_0.png)
    



```python
by_days = pd.Series(df_causal['day_week_start']).value_counts().sort_index()
by_days.index = pd.Index(by_days.index)
df_causal_days = by_days.rename_axis('days').reset_index(name='counts')
df_causal_days
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>days</th>
      <th>counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>friday</td>
      <td>686002</td>
    </tr>
    <tr>
      <th>1</th>
      <td>monday</td>
      <td>581831</td>
    </tr>
    <tr>
      <th>2</th>
      <td>saturday</td>
      <td>833136</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sunday</td>
      <td>706254</td>
    </tr>
    <tr>
      <th>4</th>
      <td>thursday</td>
      <td>621602</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tuesday</td>
      <td>615493</td>
    </tr>
    <tr>
      <th>6</th>
      <td>wednesday</td>
      <td>641247</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.pie(df_causal_days['counts'],labels=df_causal_days['days'],autopct='%1.1f%%')
plt.title('Casual')
plt.show()
```


    
![png](output_70_0.png)
    



```python
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ride_id</th>
      <th>rideable_type</th>
      <th>started_at</th>
      <th>ended_at</th>
      <th>start_station_name</th>
      <th>start_station_id</th>
      <th>end_station_name</th>
      <th>end_station_id</th>
      <th>member_casual</th>
      <th>day_week_start</th>
      <th>duration</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000001004784CD35</td>
      <td>docked_bike</td>
      <td>2020-07-22</td>
      <td>2020-07-22 15:56:47</td>
      <td>Wolcott (Ravenswood) Ave &amp; Montrose Ave</td>
      <td>238</td>
      <td>Southport Ave &amp; Clybourn Ave</td>
      <td>307</td>
      <td>casual</td>
      <td>wednesday</td>
      <td>1824.0</td>
      <td>42501.53</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000002EBE159AE82</td>
      <td>electric_bike</td>
      <td>2021-06-22</td>
      <td>2021-06-22 17:31:34</td>
      <td>Clinton St &amp; Jackson Blvd</td>
      <td>638</td>
      <td>Milwaukee Ave &amp; Grand Ave</td>
      <td>13033</td>
      <td>member</td>
      <td>tuesday</td>
      <td>619.0</td>
      <td>14982.48</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1 = pd.Series(df['rideable_type'].value_counts().sort_index())
df1.index = pd.Index(df1.index)
df1 = df1.rename_axis('rideable_type').reset_index(name = 'counts')
df1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rideable_type</th>
      <th>counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>classic_bike</td>
      <td>1205865</td>
    </tr>
    <tr>
      <th>1</th>
      <td>docked_bike</td>
      <td>2776356</td>
    </tr>
    <tr>
      <th>2</th>
      <td>electric_bike</td>
      <td>703344</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.pie(df1['counts'],labels = df1['rideable_type'], autopct='%1.1f%%')
plt.title('rideable_types for both member and casual')
plt.show()
```


    
![png](output_73_0.png)
    



```python
df2 = pd.Series(df_member['rideable_type'].value_counts())
df2.index = pd.Index(df2.index)
df2 = df2.rename_axis('rideable_type').reset_index(name = 'counts')
df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rideable_type</th>
      <th>counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>docked_bike</td>
      <td>1727733</td>
    </tr>
    <tr>
      <th>1</th>
      <td>classic_bike</td>
      <td>792599</td>
    </tr>
    <tr>
      <th>2</th>
      <td>electric_bike</td>
      <td>397107</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.pie(df2['counts'],labels = df2['rideable_type'],autopct ='%1.1f%%' )
plt.show()
```


    
![png](output_75_0.png)
    



```python
df3 = pd.Series(df_casual['rideable_type'].value_counts())
df3.index = pd.Index(df3.index)
df3 = df3.rename_axis('rideable_type').reset_index(name = 'counts')
df3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rideable_type</th>
      <th>counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>docked_bike</td>
      <td>1048623</td>
    </tr>
    <tr>
      <th>1</th>
      <td>classic_bike</td>
      <td>413266</td>
    </tr>
    <tr>
      <th>2</th>
      <td>electric_bike</td>
      <td>306237</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.pie(df3['counts'],labels = df3['rideable_type'],autopct ='%1.1f%%' )
plt.show()
```


    
![png](output_77_0.png)
    



```python

```
