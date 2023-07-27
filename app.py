from utils.deep_speech import DeepSpeech
from utils.data_processing import load_landmark_openface,compute_crop_radius
from config.config import DINetInferenceOptionsJson
from models.DINet import DINet
import zipfile
import gdown
import numpy as np
import requests
import glob
import os
import cv2
import torch
import subprocess
import base64
import random
from collections import OrderedDict


class InferlessUtils:
    @staticmethod
    def base64_to_file(base64_string, file_path):
        with open(file_path, "wb") as fh:
            fh.write(base64.decodebytes(base64_string.encode()))

    @staticmethod
    def cleanup(video_input_path, audio_input_path, video_output_path):
        os.remove(video_input_path)
        os.remove(audio_input_path)
        os.remove(video_output_path)

class InferlessPythonModel:
    def extract_frames_from_video(self, video_path,save_dir):
        videoCapture = cv2.VideoCapture(video_path)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        if int(fps) != 25:
            print('warning: the input video is not 25 fps, it would be better to trans it to 25 fps!')
        frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_width = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)
        for i in range(int(frames)):
            ret, frame = videoCapture.read()
            result_path = os.path.join(save_dir, str(i).zfill(6) + '.jpg')
            cv2.imwrite(result_path, frame)
        return (int(frame_width),int(frame_height))

    def get_confirm_token(self, response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(self, response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    def extract_zip(self, file_path, extract_path):
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        except:
            print("invalid zipfile")

    def initialize(self):
        FILE_ID = "1CkeEn7l3PuubuJIMWNjWpIrt0HDd_AB3"
        FILE_NAME = "asserts.zip"

        destination = os.path.join(os.getcwd(), FILE_NAME)

        # Download the file from Google Drive using gdown
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", destination, quiet=False)

        # Extract the downloaded zip file
        self.extract_zip(destination, os.getcwd())


    def infer(self, inputs):
        video_input = inputs["video_input"]
        audio_input = inputs["audio_input"]
        video_input_path = "video_input.mp4"
        audio_input_path = "audio_input.wav"

        InferlessUtils.base64_to_file(video_input, video_input_path)
        InferlessUtils.base64_to_file(audio_input, audio_input_path)

        inputs["source_video_path"] = video_input_path
        inputs["driving_audio_path"] = audio_input_path

        opt = DINetInferenceOptionsJson(inputs)
        if not os.path.exists(opt.source_video_path):
            raise ('wrong video path : {}'.format(opt.source_video_path))
        ############################################## extract frames from source video ##############################################
        print('extracting frames from video: {}'.format(opt.source_video_path))
        video_frame_dir = opt.source_video_path.replace('.mp4', '')
        if not os.path.exists(video_frame_dir):
            os.mkdir(video_frame_dir)
        video_size = self.extract_frames_from_video(opt.source_video_path,video_frame_dir)
        ############################################## extract deep speech feature ##############################################
        print('extracting deepspeech feature from : {}'.format(opt.driving_audio_path))
        if not os.path.exists(opt.deepspeech_model_path):
            raise ('pls download pretrained model of deepspeech')
        DSModel = DeepSpeech(opt.deepspeech_model_path)
        if not os.path.exists(opt.driving_audio_path):
            raise ('wrong audio path :{}'.format(opt.driving_audio_path))
        ds_feature = DSModel.compute_audio_feature(opt.driving_audio_path)
        res_frame_length = ds_feature.shape[0]
        ds_feature_padding = np.pad(ds_feature, ((2, 2), (0, 0)), mode='edge')
        ############################################## load facial landmark ##############################################
        print('loading facial landmarks from : {}'.format(opt.source_openface_landmark_path))
        if not os.path.exists(opt.source_openface_landmark_path):
            raise ('wrong facial landmark path :{}'.format(opt.source_openface_landmark_path))
        video_landmark_data = load_landmark_openface(opt.source_openface_landmark_path).astype(np.int)
        ############################################## align frame with driving audio ##############################################
        print('aligning frames with driving audio')
        video_frame_path_list = glob.glob(os.path.join(video_frame_dir, '*.jpg'))
        if len(video_frame_path_list) != video_landmark_data.shape[0]:
            raise ('video frames are misaligned with detected landmarks')
        video_frame_path_list.sort()
        video_frame_path_list_cycle = video_frame_path_list + video_frame_path_list[::-1]
        video_landmark_data_cycle = np.concatenate([video_landmark_data, np.flip(video_landmark_data, 0)], 0)
        video_frame_path_list_cycle_length = len(video_frame_path_list_cycle)
        if video_frame_path_list_cycle_length >= res_frame_length:
            res_video_frame_path_list = video_frame_path_list_cycle[:res_frame_length]
            res_video_landmark_data = video_landmark_data_cycle[:res_frame_length, :, :]
        else:
            divisor = res_frame_length // video_frame_path_list_cycle_length
            remainder = res_frame_length % video_frame_path_list_cycle_length
            res_video_frame_path_list = video_frame_path_list_cycle * divisor + video_frame_path_list_cycle[:remainder]
            res_video_landmark_data = np.concatenate([video_landmark_data_cycle]* divisor + [video_landmark_data_cycle[:remainder, :, :]],0)
        res_video_frame_path_list_pad = [video_frame_path_list_cycle[0]] * 2 \
                                        + res_video_frame_path_list \
                                        + [video_frame_path_list_cycle[-1]] * 2
        res_video_landmark_data_pad = np.pad(res_video_landmark_data, ((2, 2), (0, 0), (0, 0)), mode='edge')
        assert ds_feature_padding.shape[0] == len(res_video_frame_path_list_pad) == res_video_landmark_data_pad.shape[0]
        pad_length = ds_feature_padding.shape[0]

        ############################################## randomly select 5 reference images ##############################################
        print('selecting five reference images')
        ref_img_list = []
        resize_w = int(opt.mouth_region_size + opt.mouth_region_size // 4)
        resize_h = int((opt.mouth_region_size // 2) * 3 + opt.mouth_region_size // 8)
        ref_index_list = random.sample(range(5, len(res_video_frame_path_list_pad) - 2), 5)
        for ref_index in ref_index_list:
            crop_flag,crop_radius = compute_crop_radius(video_size,res_video_landmark_data_pad[ref_index - 5:ref_index, :, :])
            if not crop_flag:
                raise ('our method can not handle videos with large change of facial size!!')
            crop_radius_1_4 = crop_radius // 4
            ref_img = cv2.imread(res_video_frame_path_list_pad[ref_index- 3])[:, :, ::-1]
            ref_landmark = res_video_landmark_data_pad[ref_index - 3, :, :]
            ref_img_crop = ref_img[
                    ref_landmark[29, 1] - crop_radius:ref_landmark[29, 1] + crop_radius * 2 + crop_radius_1_4,
                    ref_landmark[33, 0] - crop_radius - crop_radius_1_4:ref_landmark[33, 0] + crop_radius +crop_radius_1_4,
                    :]
            ref_img_crop = cv2.resize(ref_img_crop,(resize_w,resize_h))
            ref_img_crop = ref_img_crop / 255.0
            ref_img_list.append(ref_img_crop)
        ref_video_frame = np.concatenate(ref_img_list, 2)
        ref_img_tensor = torch.from_numpy(ref_video_frame).permute(2, 0, 1).unsqueeze(0).float().cuda()

        ############################################## load pretrained model weight ##############################################
        print('loading pretrained model from: {}'.format(opt.pretrained_clip_DINet_path))
        model = DINet(opt.source_channel, opt.ref_channel, opt.audio_channel).cuda()
        if not os.path.exists(opt.pretrained_clip_DINet_path):
            raise ('wrong path of pretrained model weight: {}'.format(opt.pretrained_clip_DINet_path))
        state_dict = torch.load(opt.pretrained_clip_DINet_path)['state_dict']['net_g']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove module.
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.eval()
        ############################################## inference frame by frame ##############################################
        if not os.path.exists(opt.res_video_dir):
            os.mkdir(opt.res_video_dir)
        res_video_path = os.path.join(opt.res_video_dir,os.path.basename(opt.source_video_path)[:-4] + '_facial_dubbing.mp4')
        if os.path.exists(res_video_path):
            os.remove(res_video_path)
        res_face_path = res_video_path.replace('_facial_dubbing.mp4', '_synthetic_face.mp4')
        if os.path.exists(res_face_path):
            os.remove(res_face_path)
        videowriter = cv2.VideoWriter(res_video_path, cv2.VideoWriter_fourcc(*'XVID'), 25, video_size)
        videowriter_face = cv2.VideoWriter(res_face_path, cv2.VideoWriter_fourcc(*'XVID'), 25, (resize_w, resize_h))
        for clip_end_index in range(5, pad_length, 1):
            print('synthesizing {}/{} frame'.format(clip_end_index - 5, pad_length - 5))
            crop_flag, crop_radius = compute_crop_radius(video_size,res_video_landmark_data_pad[clip_end_index - 5:clip_end_index, :, :],random_scale = 1.05)
            if not crop_flag:
                raise ('our method can not handle videos with large change of facial size!!')
            crop_radius_1_4 = crop_radius // 4
            frame_data = cv2.imread(res_video_frame_path_list_pad[clip_end_index - 3])[:, :, ::-1]
            frame_landmark = res_video_landmark_data_pad[clip_end_index - 3, :, :]
            crop_frame_data = frame_data[
                                frame_landmark[29, 1] - crop_radius:frame_landmark[29, 1] + crop_radius * 2 + crop_radius_1_4,
                                frame_landmark[33, 0] - crop_radius - crop_radius_1_4:frame_landmark[33, 0] + crop_radius +crop_radius_1_4,
                                :]
            crop_frame_h,crop_frame_w = crop_frame_data.shape[0],crop_frame_data.shape[1]
            crop_frame_data = cv2.resize(crop_frame_data, (resize_w,resize_h))  # [32:224, 32:224, :]
            crop_frame_data = crop_frame_data / 255.0
            crop_frame_data[opt.mouth_region_size//2:opt.mouth_region_size//2 + opt.mouth_region_size,
                            opt.mouth_region_size//8:opt.mouth_region_size//8 + opt.mouth_region_size, :] = 0

            crop_frame_tensor = torch.from_numpy(crop_frame_data).float().cuda().permute(2, 0, 1).unsqueeze(0)
            deepspeech_tensor = torch.from_numpy(ds_feature_padding[clip_end_index - 5:clip_end_index, :]).permute(1, 0).unsqueeze(0).float().cuda()
            with torch.no_grad():
                pre_frame = model(crop_frame_tensor, ref_img_tensor, deepspeech_tensor)
                pre_frame = pre_frame.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255
            videowriter_face.write(pre_frame[:, :, ::-1].copy().astype(np.uint8))
            pre_frame_resize = cv2.resize(pre_frame, (crop_frame_w,crop_frame_h))
            frame_data[
            frame_landmark[29, 1] - crop_radius:
            frame_landmark[29, 1] + crop_radius * 2,
            frame_landmark[33, 0] - crop_radius - crop_radius_1_4:
            frame_landmark[33, 0] + crop_radius + crop_radius_1_4,
            :] = pre_frame_resize[:crop_radius * 3,:,:]
            videowriter.write(frame_data[:, :, ::-1])
        videowriter.release()
        videowriter_face.release()
        video_add_audio_path = res_video_path.replace('.mp4', '_add_audio.mp4')
        if os.path.exists(video_add_audio_path):
            os.remove(video_add_audio_path)
        cmd = 'ffmpeg -i {} -i {} -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {}'.format(
            res_video_path,
            opt.driving_audio_path,
            video_add_audio_path)
        subprocess.call(cmd, shell=True)

        video_base64 = base64.b64encode(open(video_add_audio_path, 'rb').read())
        InferlessUtils.cleanup(video_input_path, audio_input_path, video_add_audio_path)

        return {"generated_video": video_base64.decode('utf-8')}

    def finalize(self):
        pass
