import numpy as np

def sent_clip(nstory, embed, attention_dict, idim):
    fixed_num_sent = nstory

    tmp1 = np.zeros((len(embed['zsl_train']), fixed_num_sent, idim))
    i = 0
    for story in embed['zsl_train']:
        story = np.reshape(story, (-1, idim))
        story = np.reshape(story, (1, -1, idim))
        num_sent = story.shape[1]
        if num_sent > fixed_num_sent:
            tmp1[i,:,:] = story[:,:fixed_num_sent,:]
        elif num_sent < fixed_num_sent:
            padding = fixed_num_sent - num_sent
            pad_arr = np.zeros((1,padding, idim))
            tmp1[i,:num_sent,:] = story
            tmp1[i,num_sent:,:] = pad_arr
        i += 1

    tmp2 = np.zeros((len(embed['zsl_val']), fixed_num_sent, idim))
    i = 0
    for story in embed['zsl_val']:
        story = np.reshape(story, (-1, idim))
        story = np.reshape(story, (1, -1, idim))
        num_sent = story.shape[1]
        if num_sent > fixed_num_sent:
            tmp2[i,:,:] = story[:,:fixed_num_sent,:]
        elif num_sent < fixed_num_sent:
            padding = fixed_num_sent - num_sent
            pad_arr = np.zeros((1,padding, idim))
            tmp2[i,:num_sent,:] = story
            tmp2[i,num_sent:,:] = pad_arr
        i += 1

    tmp3 = np.zeros((len(attention_dict['train']), fixed_num_sent, 1))
    i = 0
    for story in attention_dict['train']:
        story = np.reshape(story, (-1, 1)) # [num_story, 1]
        story = np.reshape(story, (1, -1, 1)) # [1, num_story ,1]
        num_sent = story.shape[1]
        if num_sent > fixed_num_sent:
            tmp3[i,:,:] = story[:,:fixed_num_sent,:]
        elif num_sent < fixed_num_sent:
            padding = fixed_num_sent - num_sent
            pad_arr = np.zeros((1, padding, 1))
            tmp3[i,:num_sent,:] = story
            tmp3[i,num_sent:,:] = pad_arr
        i += 1

    tmp4 = np.zeros((len(attention_dict['val']), fixed_num_sent, 1))
    i = 0
    for story in attention_dict['val']:
        story = np.reshape(story, (-1, 1)) # [num_story, 1]
        story = np.reshape(story, (1, -1, 1)) # [1, num_story ,1]
        num_sent = story.shape[1]
        if num_sent > fixed_num_sent:
            tmp4[i,:,:] = story[:,:fixed_num_sent,:]
        elif num_sent < fixed_num_sent:
            padding = fixed_num_sent - num_sent
            pad_arr = np.zeros((1, padding, 1))
            tmp4[i,:num_sent,:] = story
            tmp4[i,num_sent:,:] = pad_arr
        i += 1

    print "clipped shape (train) >> ",
    print tmp1.shape
    print "clipped shape (val) >> ",
    print tmp2.shape
    print 'clipped attention (train) >> ',
    print tmp3.shape
    print 'clipped attention (val) >> ',
    print tmp4.shape

    return tmp1, tmp2, tmp3, tmp4
