# Profling Report For mindspore_glLearning Against DGL/PyG/PGL


## Sampling Profile

### Environment Setting

|H/W path           Device|       Class|          Description|
|----|----|----|
|      |                          system  |       Computer|
|/0 |                             bus     |       Motherboard|
|/0/6 |                           memory   |      251GiB System memory|
|/0/7  |                          processor |     Intel(R) Xeon(R) CPU E5-2660 v4 @ 2.00GHz |
|/0/8   |                         processor  |    Intel(R) Xeon(R) CPU E5-2660 v4 @ 2.00GHz|
|/0/100  |                        bridge     |    Xeon E7 v4/Xeon E5 v4/Xeon E3 v4/Xeon D DMI2 |
|/0/100/1 |                       bridge      |   Xeon E7 v4/Xeon E5 v4/Xeon E3 v4/Xeon D PCI Express Root Port 1 |
|/0/100/1/0|                      storage      |  MegaRAID SAS 2208 [Thunderbolt]|

### Profile Result

**SAGE Sampling**

|参数|值|
|----|----|
|BatchSize|1024|
|Neighbor Sampling Parameter|[25, 10]|
|Replaceable|False|

|框架|采样时间消耗(ms)|实现方式|
|---|---|---|
|PyG|251|C++|
|MGL|310|Cython|