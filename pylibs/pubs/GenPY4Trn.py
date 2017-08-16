#!D:/Python27
#coding:UTF-8
# -*- coding: utf-8 -*-
#author :lyi 20161230

import os
import sys 
import linecache
import string
import codecs
import shutil
#import json
#import pylab as pl

reload(sys)
sys.setdefaultencoding('utf-8')


#定义初加工文件每行的信息
class PreProcessInfo(object):
    def __init__(self, begintime,endtime,sex,role,label):
        self.begintime=begintime
        self.endtime  =endtime
        self.sex      =sex
        self.role      =role
        self.label      =label
#定义原始wav文件名信息
class FileNameInfo(object):
    def __init__(self, channel,date):
        self.channel=channel
        self.date  =date
#定义样本trn文件信息
class TrnObjInfo(object):
    def __init__(self, zhs,py,sym,bm,be):
        self.zhs=zhs#汉字
        self.py  =py#拼音
        self.sym=sym#声母、韵母
        self.bm=bm#包含的多音字
        self.be=be#是否包含错误
    
#声母数组
Initials=[];
#韵母数组
Vowels=[];
 #映射表
PairMap = {};
#通道编号
ChanelMap={};
#自定义的词组对应拼音
WordsMap={};
#unicode和汉语映射
dictUnicode=dict();
#词组unicode
WordUC=[];
#这些都是单韵母
Special=["a","e","i","o"];
g_ppi = PreProcessInfo(100,8,"m","c","h") 
g_fni=FileNameInfo("01","20161201")
g_tbi=TrnObjInfo("","","","",0)
#读取声母韵母转换配置参数
def ReadParas(cfgpath):
    #读取声母表
    #配置文件路径
    #第一行为声母，第二行为韵母，第三行为空格，接下来是声母韵母的样本
    #格式一定要正确
    path_ini=cfgpath+"Pair.map";
    if not os.path.exists(path_ini):
        print "parameters file Pair.map is not exist in current dir."
    oneline = linecache.getline(path_ini,1).strip();
    oneline=oneline.replace("\t"," ");
    oneline=oneline.replace("\n","");
    Initials = oneline.split(" ");
    #读取韵母表
    oneline = linecache.getline(path_ini,2).strip();
    oneline=oneline.replace("\t"," ");
    oneline=oneline.replace("\n","");
    Vowels = oneline.split(" ");
    
    print "Inis",len(Initials);
    print "Vows",len(Vowels);
    
    #读取映射表
    iLine=0;
    f=open(path_ini,'r')
    for line in f:
        line=line.replace("\n","").replace('\r','');
        iLine = iLine + 1;
        #print(iLine,line)
        if line == '' or iLine <= 3:
            continue;        
        all_words=line.split("\t");
        #print all_words,len(all_words);
        sindex=iLine-5;
        for i in range(len(all_words)):
            yindex=i;
            #print(sindex,yindex)
            key=Initials[sindex]+Vowels[yindex];
            value=all_words[i];
            value=value.replace("-"," ");
            PairMap[key]=value;
            #print key,value;
    f.close()
    
    #print "Sample Pair",PairMap;

#读取自定义词组列表
def ReadWordDecode(cfgpath):
    path_ini=cfgpath+"multi-tone-phrase.dict";
    if not os.path.exists(path_ini):
        print "parameters words.dat is not exist in current dir."            
    f=codecs.open(path_ini,'r','utf-8')
    for line in f:
        if line.startswith('#'):
            continue;
        segs=line.split("|");
        if len(segs) > 1:
            skey= segs[0]#.decode('utf-8','ignore');
            WordsMap[ skey ]=segs[1].strip();    
    f.close();
    print("WordsMap",len(WordsMap))

#read unicode和汉语拼音映射
def ReadUnicodePara(cfgpath):
    path_ini=cfgpath+"unicode_to_hanyu_pinyin.txt";
    if not os.path.exists(path_ini):
        print "parameters unicode_to_hanyu_pinyin.txt is not exist in current dir."                
    f=open(path_ini,'r')
    for line in f:
        str=line.strip().split(' ');
        uc=str[0].strip();
        py=str[1].strip().replace('(','').replace(')','');
        if "," in py:
            str=py.split(",");
            for sp in str:
                if dictUnicode.has_key(sp):
                    dictUnicode[uc].append(sp);
                else:
                    dictUnicode[uc]= py;
        else:
            dictUnicode[uc]= py;        
    f.close();
#    print dictUnicode;



#解析初加工文件中每一行信息    
def ParseLine(line)    :
    line = line.strip('\xef\xbb\xbf')#BOM
    seg1=line.split('|')
    if len(seg1) > 1:
        g_ppi.label=seg1[1];
        #print g_ppi.label;
    seg2=seg1[0].split(' ')
    if len(seg2) > 3:
         print seg2
         g_ppi.begintime=string.atoi(seg2[0],10)
         g_ppi.endtime=string.atoi(seg2[1],10)
         g_ppi.sex=seg2[2]
         g_ppi.role=seg2[3]
#解析wav文件名信息    
def ParseSecDataInfo(wav_file_name):
    seg=wav_file_name;
    seg=seg.replace('.wav','')
    seg=wav_file_name.split("_");
    s=seg[0];
    g_fni.channel="";
    g_fni.date="";
    for one_ch in s:
        intord = ord(one_ch)
        #数字
        if (intord >= 48 and intord <= 57):    
            g_fni.channel = g_fni.channel+one_ch;
    #print g_fni.channel;
    s=seg[-1];
    g_fni.date=s[0:8];
#时间转换，001212250转换为秒，小数点后为毫秒
def trans_time(i_time):
    s='%d' %i_time;
    s=s.zfill(9);
    h=string.atoi(s[0:2],10);
    m=string.atoi(s[2:4],10);
    s=string.atoi(s[4:6],10);
    ms=i_time%1000;
    irtn=h*3600+m*60+s+float(ms)/1000.0 ;
    return irtn;


#获取声母韵母,可以为多个拼音，以空格隔开
def GetSYM(py):
    sym="";
    if py=="":
        return sym;
    all_py = py.split(" ");
    for one_py in all_py:
        shengdiao=one_py[-1];
        pinyin=one_py[0:len(one_py)-1];
        sym =  sym + " "+PairMap.get(pinyin, pinyin)+shengdiao;
    return sym;

def GetUnicode(one_ch):
    uc= one_ch.decode('utf-8','ignore');#unicode_escape
    uc=repr(uc.decode('utf-8','ignore'));
    uc=uc.upper();
    uc=uc[4:8];
    return uc;

def GetWordsUC(one_ch):
    uc= one_ch.decode('utf-8','ignore');#unicode_escape
    uc=repr(uc.decode('utf-8','ignore'));
    return uc;
    
def test(s):
	return ProcLabel(s)
	
def ProcLabel(label_zhs):
    g_tbi.zhs=label_zhs;
    g_tbi.bm="";
    g_tbi.be=0;
    g_tbi.py="";
    g_tbi.sym="";
    #这里可能需要修改，lyi
    one_len = 1;#len("柯") #测试汉字占用字节数，utf-8，汉字占用3字节.bg2312，汉字占用2字节
    
    
    #利用初加工样本进行分词处理
    all_seg_words = label_zhs.split(" ");
    #print all_seg_words;
    for words in all_seg_words:
        #首先直接在自定义词组表中查询
        words=words.strip()
        print words#,  WordsMap.keys()[1]
        #worduc = '校飞'.decode('utf-8','ignore')
        if words in WordsMap:
            this_py=WordsMap[words]#.get(wuc, "");            
            print "in words[",this_py,"]";
            g_tbi.py=g_tbi.py+" " +this_py;
            sym = GetSYM(this_py);
            g_tbi.sym=g_tbi.sym+" " +sym;
        #否则进行单个字处理
        else:
            ch=words;#.replace(" ","");
            ch=ch.replace("\n","");
            ch=ch.replace("\r","");
            all_len=len(ch);
            print ch,all_len;
            iNum=1;
            while iNum*one_len <= all_len:
                one_ch = ch[(iNum-1)*one_len:iNum*one_len] #取字
                iNum=iNum+1;
                if one_ch=="" or one_ch==" ":
                    continue;
                '''
                intord = ord(one_ch)
                #数字
                if (intord >= 48 and intord <= 57):
                    g_tbi.py=g_tbi.py+" " +one_ch;
                #字母
                if (intord >= 65 and intord <=90 ) or (intord >= 97 and intord <=122):
                    g_tbi.py=g_tbi.py+" " +one_ch;
                '''
                #备注，正确情况下应该只有汉字
                #汉字
                wl=len(one_ch)
                if wl >= one_len:
                    #下面这几步必须处理，不然找不到对应的拼音
                    #print json.load("%s"%one_ch)
#                    uc=repr(one_ch.decode('GBK','ignore'));
                    #uc=one_ch.decode('utf-8')#unicode(one_ch,'utf-8')#repr(one_ch.decode('GBK'));
                    uc=GetUnicode( one_ch)
                    '''
                    .decode('utf-8','ignore');#unicode_escape
                    uc=repr(uc.decode('utf-8','ignore'));
                    uc=uc.upper();
                    uc=uc[4:8];
                    '''
                    if dictUnicode.has_key(uc):#利用Unicode取拼音
                        this_pys=dictUnicode.get(uc).split(",")
                        this_py=this_pys[0]
                        #多音字，默认取第一个
                        #if len(this_pys) > 1:
                        #    this_py=this_pys[0];
                        #print this_py;
                        g_tbi.py=g_tbi.py+" " +this_py
                        sym = GetSYM(this_py)
                        g_tbi.sym=g_tbi.sym+" " +sym
                    else:
                        g_tbi.be=1;#错误
    
    #删除句首和句尾的空格
    g_tbi.zhs=g_tbi.zhs.strip()
    g_tbi.py=g_tbi.py.strip()
    g_tbi.sym=g_tbi.sym.strip()
    print(g_tbi.py,g_tbi.sym)
    return  label_zhs + label_zhs,g_tbi.py,g_tbi.sym
 
from cmudict import CMUdict                  
class GenPY(object):
    def __init__(self, cfgpath,cmupath):
        if not cfgpath.endswith('/'):
            cfgpath+='/'
        ReadParas(cfgpath)
        ReadUnicodePara(cfgpath)
        ReadWordDecode(cfgpath)
        self.cmuproc=CMUdict(cmupath)
        #print(PairMap)
    def ProcLine(self,label_zhs):
        PY,SYM='',''
        all_seg_words = label_zhs.split(" ");
        #print all_seg_words;
        for words in all_seg_words:
            words=words.strip()
            try:
                ucs=unicode(words,'utf-8')
            except:
                continue
            #print words,ucs#,  WordsMap.keys()[1]
            if ucs in WordsMap:
                this_py=WordsMap[ucs]#.get(wuc, "");            
                print "in words[",this_py,"]";
                PY=PY+" " +this_py;
                sym = GetSYM(this_py);
                SYM=SYM+" " +sym;
            elif words.isalpha():
                en_phone=self.cmuproc.phones_for_sentence(words)
                PY=PY+" "+en_phone.replace(' ','')
                SYM=SYM+" "+en_phone
            elif words.isdigit():
                PY=PY+" "+words
                SYM=SYM+" "+words
            #否则进行单个字处理
            else:
                for uc in ucs:
                    uc=GetUnicode(uc)
                    if dictUnicode.has_key(uc):
                        this_pys=dictUnicode.get(uc).split(",")
                        this_py=this_pys[0]
                        PY=PY+" " +this_py;
                        sym = GetSYM(this_py);
                        SYM=SYM+" " +sym;
                    else:
                        PY=PY+" "+words
                        SYM=SYM+" "+words
        SYM=SYM.strip().replace('  ',' ')
        PY=PY.strip().replace('  ',' ')
        #print('result:',PY,SYM)        
        return PY,SYM
    def GetPY(self,line):
        #print line
        line = line.strip('\xef\xbb\xbf')#BOM
        return self.ProcLine(line)

if __name__ == '__main__':
    s='下 alpha'
    genPhone=GenPY(cfgpath='../resource/chn_py',cmupath='../resource/cmu/cmudict-0.7b')
    print(genPhone.ProcLine(s))
    exit()
    if len(sys.argv) != 3:
        sys.stderr.write('Usage: python GenPY4Trn.py <inputfile> <outputfile>\n')
        sys.exit(1)
    
     

    #read parameters
    ReadParas()#读取声母韵母对应表
    ReadUnicodePara()#读取汉字Unicode与拼音对应表
    ReadWordDecode()#读取自定义单词与拼音对应表

    #read all data which should be processed
    #print dictUnicode.get("4E07");
    #print PairMap["wo"];

    #print WordsMap;
    
    #f = codecs.open(sys.argv[1],'r','utf-8')
    #line = f.readline()

    line = sys.argv[1].decode('gbk')
    #print line
    line = line.strip('\xef\xbb\xbf')#BOM
    ProcLabel(line)
    fw=open(sys.argv[2],'w')
    print g_tbi.zhs
    print g_tbi.py
    print g_tbi.sym
    fw.write(g_tbi.zhs+"\n"+g_tbi.py+"\n"+g_tbi.sym+"\n")
    fw.close()
