import numpy as np
import xml.dom.minidom as minidom

def write_xml(xml_name,data_name,image_shape,boxes):
    
    doc=minidom.Document()
    root=doc.createElement("annotation")
    doc.appendChild(root)

    filename = doc.createElement('filename')
    filename_text = doc.createTextNode(data_name)
    filename.appendChild(filename_text)
    root.appendChild(filename)

    database = doc.createElement('database')
    database_text = doc.createTextNode('kitti')
    database.appendChild(database_text)
    root.appendChild(database)

    size = doc.createElement('size')

    width = doc.createElement('width')
    width_text = doc.createTextNode('%d'%image_shape[1])
    width.appendChild(width_text)
    size.appendChild(width)

    height = doc.createElement('height')
    height_text = doc.createTextNode('%d'%image_shape[0])
    height.appendChild(height_text)
    size.appendChild(height)

    depth = doc.createElement('depth')
    depth_text = doc.createTextNode('%d'%image_shape[2])
    depth.appendChild(depth_text)
    size.appendChild(depth)

    root.appendChild(size)

    for bbox in boxes:
        object_ = doc.createElement('object')

        name = doc.createElement('name')
        name_text = doc.createTextNode('car')
        name.appendChild(name_text)
        object_.appendChild(name)

        difficult = doc.createElement('difficult')
        difficult_text = doc.createTextNode('0')
        difficult.appendChild(difficult_text)
        object_.appendChild(difficult)

        truncated = doc.createElement('truncated')
        truncated_text = doc.createTextNode('%f'%bbox["truncated"])
        truncated.appendChild(truncated_text)
        object_.appendChild(truncated)

        occluded = doc.createElement('occluded')
        occluded_text = doc.createTextNode('%d'%bbox["occluded"])
        occluded.appendChild(occluded_text)
        object_.appendChild(occluded)

        #----left box-----
        left_bndbox = doc.createElement('left_bndbox')
        
        xmin = doc.createElement('xmin')
        xmin_text = doc.createTextNode('%f'%bbox["left_box"][0])
        xmin.appendChild(xmin_text)
        left_bndbox.appendChild(xmin)

        ymin = doc.createElement('ymin')
        ymin_text = doc.createTextNode('%f'%bbox["left_box"][1])
        ymin.appendChild(ymin_text)
        left_bndbox.appendChild(ymin)

        xmax = doc.createElement('xmax')
        xmax_text = doc.createTextNode('%f'%bbox["left_box"][2])
        xmax.appendChild(xmax_text)
        left_bndbox.appendChild(xmax)

        ymax = doc.createElement('ymax')
        ymax_text = doc.createTextNode('%f'%bbox["left_box"][3])
        ymax.appendChild(ymax_text)
        left_bndbox.appendChild(ymax)

        center = doc.createElement('center')

        center_x = doc.createElement('x')
        center_x_text = doc.createTextNode('%f'%bbox["center_left"][0])
        center_x.appendChild(center_x_text)
        center.appendChild(center_x)
        center_y = doc.createElement('y')
        center_y_text = doc.createTextNode('%f'%bbox["center_left"][1])
        center_y.appendChild(center_y_text)
        center.appendChild(center_y)

        left_bndbox.appendChild(center)

        object_.appendChild(left_bndbox)

        
        #----right box-----
        right_bndbox = doc.createElement('right_bndbox')
        
        xmin = doc.createElement('xmin')
        xmin_text = doc.createTextNode('%f'%bbox["right_box"][0])
        xmin.appendChild(xmin_text)
        right_bndbox.appendChild(xmin)

        ymin = doc.createElement('ymin')
        ymin_text = doc.createTextNode('%f'%bbox["right_box"][1])
        ymin.appendChild(ymin_text)
        right_bndbox.appendChild(ymin)

        xmax = doc.createElement('xmax')
        xmax_text = doc.createTextNode('%f'%bbox["right_box"][2])
        xmax.appendChild(xmax_text)
        right_bndbox.appendChild(xmax)

        ymax = doc.createElement('ymax')
        ymax_text = doc.createTextNode('%f'%bbox["right_box"][3])
        ymax.appendChild(ymax_text)
        right_bndbox.appendChild(ymax)

        center = doc.createElement('center')

        center_x = doc.createElement('x')
        center_x_text = doc.createTextNode('%f'%bbox["center_right"][0])
        center_x.appendChild(center_x_text)
        center.appendChild(center_x)
        center_y = doc.createElement('y')
        center_y_text = doc.createTextNode('%f'%bbox["center_right"][1])
        center_y.appendChild(center_y_text)
        center.appendChild(center_y)

        right_bndbox.appendChild(center)

        object_.appendChild(right_bndbox)
        

        #---position---
        pos = doc.createElement('position')
        
        x = doc.createElement('x')
        x_text = doc.createTextNode('%f'%bbox["positions"][0])
        x.appendChild(x_text)
        pos.appendChild(x)

        y = doc.createElement('y')
        y_text = doc.createTextNode('%f'%bbox["positions"][1])
        y.appendChild(y_text)
        pos.appendChild(y)

        z = doc.createElement('z')
        depth = doc.createElement('depth')
        depth_text = doc.createTextNode('%f'%bbox["positions"][2])
        depth.appendChild(depth_text)
        z.appendChild(depth)

        disp = doc.createElement('disp')
        disp_text = doc.createTextNode('%f'%bbox["disp"])
        disp.appendChild(disp_text)
        z.appendChild(disp)
        pos.appendChild(z)
        
        object_.appendChild(pos)
        
        '''
        #ph laji hahaha...
        z = doc.createElement('z')
        z_text = doc.createTextNode('%f'%bbox["positions"][2])
        z.appendChild(z_text)
        pos.appendChild(z)
        
        object_.appendChild(pos)
        
        disp = doc.createElement('disp')
        disp_text = doc.createTextNode('%f'%bbox["disp"])
        disp.appendChild(disp_text)
        object_.appendChild(disp)
        '''
        #-----dim--------
        dim = doc.createElement('dimensions')

        h = doc.createElement('h')
        h_text = doc.createTextNode('%f'%bbox["dimensions"][1])
        h.appendChild(h_text)
        dim.appendChild(h)

        w = doc.createElement('w')
        w_text = doc.createTextNode('%f'%bbox["dimensions"][0])
        w.appendChild(w_text)
        dim.appendChild(w)

        l = doc.createElement('l')
        l_text = doc.createTextNode('%f'%bbox["dimensions"][2])
        l.appendChild(l_text)
        dim.appendChild(l)

        object_.appendChild(dim)

        #-----other--------
        al = doc.createElement('alpha')
        al_text = doc.createTextNode('%f'%bbox["alpha"])
        al.appendChild(al_text)
        object_.appendChild(al)

        rot = doc.createElement('rotation')
        rot_text = doc.createTextNode('%f'%bbox["rotation"])
        rot.appendChild(rot_text)
        object_.appendChild(rot)

        corner = doc.createElement('corners')
        for j in range(8):
            pc = doc.createElement('pc%d'%j)
            pc_text= doc.createTextNode(bbox['point_coner'][j])
            pc.appendChild(pc_text)
            corner.appendChild(pc)
        
        object_.appendChild(corner)

        root.appendChild(object_)

    with open(xml_name,'w') as xml_file:
        doc.writexml(xml_file,indent = '\t',newl = '\n', addindent = '\t',encoding='utf-8')
