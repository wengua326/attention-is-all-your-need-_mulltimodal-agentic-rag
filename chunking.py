from unstructured.partition.pdf import partition_pdf
file_path='./content/attention.pdf'

chunks=partition_pdf(
    filename=file_path,
    infer_table_structure=True, #extract table
    strategy='hi_res',

    extract_image_block_types=['Image'], #add table to the list to extract image of table
    extract_image_block_to_payload=True,

    chunking_strategy="by_title", #根据大标题切，如果用basic 的话跟字数切
    max_characters=10000, #de3fault 500
    combine_text_under_n_chars=2000, #default 0,如果切出来<2000字就combine下一个
    new_after_n_chars=6000,

)

#separate table from text 分类
#chunk有两个类型，一个table，另一个CompositeElement
# CompositeElement里面有text,title,image
#metadata是chunk的信息， .orig_element 是专属CompositeElement的，是那三种原始数据的信息， .image_base64 是image的信息

tables=[]
texts=[]
for chunk in chunks:
    if 'table' in str(type(chunk)):
        tables.append(chunk)
    if'CompositeElement' in str(type(chunk)):
        texts.append(chunk) #之所以不要chunk.text是因为会丢失掉chunk.metadata,只剩下text罢了

#separate image
def getImage(chunks):
    Image=[]
    for chunk in chunks:
        if 'CompositeElement' in str(type(chunk)):
            chunk_data=chunk.metadata.orig_elements   #.orig_elements 是一个存3种原始数据的list
            for data in chunk_data:
                if 'Image' in str(type(data)):
                    Image.append(data) #这个data还不是最终的image信息
    return Image

images=getImage(chunks)



