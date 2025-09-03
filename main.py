import io
import numpy as np
import streamlit as st
import librosa
import soundfile as sf
import plotly.graph_objects as go
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from datetime import datetime, timedelta
from japanese_classes import get_japanese_class_name

# -----------------------------
# 定数
# -----------------------------
TARGET_SR = 16000  # YAMNetは16 kHzモノ
DEFAULT_WINDOW_SEC = 0.96   # YAMNetのフレーム長
DEFAULT_HOP_SEC = 0.48      # YAMNetのフレームホップ

# クラスラベル取得（TF公式のクラスマップCSVをキャッシュ保存）
LABELS_URL = "https://storage.googleapis.com/audioset/yamnet/yamnet_class_map.csv"

@st.cache_resource
def load_yamnet():
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    import csv
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    class_names = []
    with tf.io.gfile.GFile(class_map_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_names.append(row["display_name"])
    return model, class_names


def load_csv_audio(csv_file, target_date=None):
    """CSVファイルから時系列音声データを読み込み"""
    # ファイルストリームの場合は先頭に戻す
    if hasattr(csv_file, 'seek'):
        csv_file.seek(0)
    
    df = pd.read_csv(csv_file)
    
    # マイクロ秒を秒に変換
    timestamps = df['absolute_time_us'] / 1000000.0
    amplitudes = df['amplitude'].values
    
    # タイムスタンプの形式をチェックして適切に処理
    # 非常に大きな値の場合は、相対時間として扱う
    if len(timestamps) > 0:
        first_timestamp = timestamps.iloc[0] if hasattr(timestamps, 'iloc') else timestamps[0]
        last_timestamp = timestamps.iloc[-1] if hasattr(timestamps, 'iloc') else timestamps[-1]
        
        # 通常のUnixタイムスタンプ（1970年以降）の場合は1,000,000,000秒以上
        # それより小さい場合は相対時間として処理
        if first_timestamp < 1000000000:
            # 相対時間の場合：現在時刻を基準にして絶対時間を生成
            current_time = datetime.now()
            base_timestamp = current_time.timestamp()
            # 最初のタイムスタンプを0秒として、相対的な時間差を保持
            relative_duration = last_timestamp - first_timestamp
            timestamps = timestamps - first_timestamp + base_timestamp
            
            print(f"情報: 相対時間データとして処理（継続時間: {relative_duration:.2f}秒）")
    
    # 日付フィルタリング（指定された場合）
    if target_date is not None:
        # 日付範囲を計算（指定日の00:00:00から23:59:59まで）
        start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        start_timestamp = start_of_day.timestamp()
        end_timestamp = end_of_day.timestamp()
        
        # 指定日のデータのみを抽出
        date_mask = (timestamps >= start_timestamp) & (timestamps <= end_timestamp)
        if date_mask.sum() == 0:
            # 指定日にデータがない場合は空のデータを返す
            return np.array([]), TARGET_SR, start_of_day, end_of_day, np.array([])
        
        timestamps = timestamps[date_mask]
        amplitudes = amplitudes[date_mask]
        
        # インデックスをリセット
        timestamps = timestamps.reset_index(drop=True) if hasattr(timestamps, 'reset_index') else pd.Series(timestamps).reset_index(drop=True)
        # amplitudes は既に date_mask でフィルタリング済みなので、reset_index のみ実行
        if hasattr(amplitudes, 'reset_index'):
            amplitudes = amplitudes.reset_index(drop=True)
        elif isinstance(amplitudes, np.ndarray):
            # numpy配列の場合は何もしない（すでにフィルタリング済み）
            pass
    
    # 無効な値（NaN、inf）をクリーニング
    valid_mask = np.isfinite(amplitudes)
    if not valid_mask.all():
        print(f"警告: {(~valid_mask).sum()} 個の無効な値を修正しました")
        amplitudes = np.where(valid_mask, amplitudes, 0.0)
    
    # サンプリング周波数を推定（最初の数サンプルから）
    if len(timestamps) > 1:
        dt = np.mean(np.diff(timestamps[:min(100, len(timestamps))]))  # 最初の100サンプル（またはそれ以下）の平均間隔
        estimated_sr = int(1.0 / dt) if dt > 0 else TARGET_SR
        
        # サンプル数から実際の継続時間を計算（タイムスタンプが不正確な場合のフォールバック）
        calculated_duration = len(amplitudes) / estimated_sr
        timestamp_duration = timestamps.iloc[-1] - timestamps.iloc[0] if hasattr(timestamps, 'iloc') else timestamps[-1] - timestamps[0]
        
        # タイムスタンプから計算した継続時間が異常に長い場合、サンプル数ベースを使用
        if timestamp_duration > calculated_duration * 2 or timestamp_duration < calculated_duration / 2:
            print(f"警告: タイムスタンプが不正確です。サンプル数から継続時間を計算しています。")
            print(f"  タイムスタンプベース: {timestamp_duration:.2f}秒")
            print(f"  サンプル数ベース: {calculated_duration:.2f}秒")
            
            # タイムスタンプを修正：線形に再配置
            first_timestamp = timestamps.iloc[0] if hasattr(timestamps, 'iloc') else timestamps[0]
            timestamps = pd.Series(np.linspace(first_timestamp, first_timestamp + calculated_duration, len(timestamps)))
    else:
        estimated_sr = TARGET_SR
    
    # 振幅を正規化（-1.0 to 1.0）
    if len(amplitudes) > 0 and amplitudes.max() != amplitudes.min():
        amplitudes = 2.0 * (amplitudes - amplitudes.min()) / (amplitudes.max() - amplitudes.min()) - 1.0
    else:
        amplitudes = np.zeros_like(amplitudes)  # 全て同じ値の場合は0にする
    
    # 最終チェック: まだ無効な値がある場合は0にする
    amplitudes = np.where(np.isfinite(amplitudes), amplitudes, 0.0)
    
    # 開始時刻と終了時刻を計算
    if len(timestamps) > 0:
        start_time = datetime.fromtimestamp(timestamps.iloc[0] if hasattr(timestamps, 'iloc') else timestamps[0])
        end_time = datetime.fromtimestamp(timestamps.iloc[-1] if hasattr(timestamps, 'iloc') else timestamps[-1])
    else:
        start_time = datetime.now()
        end_time = datetime.now()
    
    return amplitudes.astype(np.float32), estimated_sr, start_time, end_time, timestamps

def ensure_mono_16k(y, sr):
    # モノ化
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    # 16kへリサンプル
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    # YAMNetはfloat32想定
    return y.astype(np.float32)

def slice_by_datetime(y, timestamps, start_datetime, duration_sec):
    """指定した日時から音声区間を抽出"""
    start_timestamp = start_datetime.timestamp()
    end_timestamp = start_timestamp + duration_sec
    
    # タイムスタンプ範囲内のインデックスを取得
    mask = (timestamps >= start_timestamp) & (timestamps <= end_timestamp)
    indices = np.where(mask)[0]
    
    if len(indices) == 0:
        # 該当する時間範囲がない場合は最も近い時刻から抽出
        closest_idx = np.argmin(np.abs(timestamps - start_timestamp))
        segment_length = int(duration_sec * TARGET_SR)
        start_idx = max(0, closest_idx)
        end_idx = min(len(y), start_idx + segment_length)
        segment = y[start_idx:end_idx]
    else:
        segment = y[indices]
    
    # 必要に応じてパディング
    target_length = int(duration_sec * TARGET_SR)
    if len(segment) < target_length:
        segment = np.pad(segment, (0, target_length - len(segment)), mode="constant")
    elif len(segment) > target_length:
        segment = segment[:target_length]
    
    return segment

def slice_by_time(y, start_sec, duration_sec, sr=TARGET_SR):
    start = int(start_sec * sr)
    end = int((start_sec + duration_sec) * sr)
    start = max(0, start)
    end = min(len(y), end)
    segment = y[start:end]
    # 必要に応じてパディング（短すぎる場合でも推論可能に）
    need = int(duration_sec * sr) - len(segment)
    if need > 0:
        segment = np.pad(segment, (0, need), mode="constant")
    return segment

def aggregate_scores(scores, method="mean"):
    # scores: (num_frames, num_classes)
    if method == "mean":
        return scores.mean(axis=0)
    elif method == "max":
        return scores.max(axis=0)
    else:
        return scores.mean(axis=0)

def make_wave_plot(y, sr, start_sec, duration_sec, timestamps=None, start_datetime=None):
    # 長い音声をダウンサンプリングして表示
    max_points = 10000
    if len(y) > max_points:
        step = len(y) // max_points
        y_display = y[::step]
        if timestamps is not None:
            t_display = timestamps[::step]
            if start_datetime:
                # 日時表示
                t_display = [datetime.fromtimestamp(ts) for ts in t_display]
                x_title = "日時"
            else:
                t_display = t_display - t_display[0]  # 相対時間
                x_title = "時間 [秒]"
        else:
            t_display = np.arange(len(y_display)) * step / sr
            x_title = "時間 [秒]"
    else:
        y_display = y
        if timestamps is not None:
            t_display = timestamps
            if start_datetime:
                t_display = [datetime.fromtimestamp(ts) for ts in t_display]
                x_title = "日時"
            else:
                t_display = t_display - t_display[0]
                x_title = "時間 [秒]"
        else:
            t_display = np.arange(len(y)) / sr
            x_title = "時間 [秒]"
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_display, y=y_display, mode="lines", name="waveform"))
    fig.update_layout(
        xaxis_title=x_title, yaxis_title="振幅",
        margin=dict(l=40, r=10, t=30, b=40), height=250
    )
    
    # 選択区間の表示（ハイライト）
    if timestamps is not None and start_datetime:
        end_datetime = start_datetime + timedelta(seconds=duration_sec)
        fig.add_vrect(x0=start_datetime, x1=end_datetime, opacity=0.2, line_width=0)
    else:
        fig.add_vrect(x0=start_sec, x1=start_sec + duration_sec, opacity=0.2, line_width=0)
    
    return fig

def make_scores_heatmap(frame_times, scores, topk_idx, class_names=None):
    # 上位クラスの時系列スコアのみヒートマップ表示
    sel_scores = scores[:, topk_idx]  # (num_frames, topk)
    
    # クラス名を日本語で表示
    if class_names is not None:
        y_labels = [get_japanese_class_name(class_names[i]) for i in topk_idx]
    else:
        y_labels = [str(i) for i in topk_idx]
    
    fig = go.Figure(
        data=go.Heatmap(
            z=sel_scores.T,
            x=frame_times,
            y=y_labels,
            colorbar=dict(title="スコア")
        )
    )
    fig.update_layout(
        xaxis_title="時間 [秒]", 
        yaxis_title="音響クラス（Top-K）",
        height=250, margin=dict(l=40, r=10, t=30, b=40)
    )
    return fig

def run_yamnet(model, waveform):
    # model(audio) → (scores, embeddings, spectrogram)
    # scores shape: [num_frames, 521]
    scores, embeddings, spectrogram = model(waveform)
    # numpyへ
    return scores.numpy(), embeddings.numpy(), spectrogram.numpy()

def yamnet_frame_times(n_frames, sr=TARGET_SR, hop_sec=DEFAULT_HOP_SEC):
    # フレーム中心時刻の近似（0.48 s ホップで増える）
    return np.arange(n_frames) * hop_sec

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="YAMNet クイック可視化&区間分類", layout="wide")
st.title("YAMNet：波形の区間指定で音種を推定")

def get_window_duration(duration_type, manual_value=None):
    """区間長の種類に応じて秒数を計算"""
    if duration_type == "短め2秒":
        return 2.0  # 2秒
    elif duration_type == "推奨5秒":
        return 5.0  # 5秒
    elif duration_type == "長め10秒":
        return 10.0  # 10秒
    elif duration_type == "マニュアル":
        return manual_value if manual_value is not None else 5.0
    else:
        return 5.0  # デフォルト

def get_accuracy_warning(duration_sec):
    """区間長に応じた精度警告メッセージを生成"""
    if duration_sec < 1.0:
        return "⚠️ 警告: 1秒未満では識別精度が非常に低くなります"
    elif duration_sec < 2.0:
        return "⚠️ 注意: 2秒未満では識別精度が低下する可能性があります"
    elif duration_sec < 3.0:
        return "⚡ 情報: 最短区間のため精度にばらつきがあります"
    elif duration_sec >= 5.0:
        return "✅ 良好: 安定した識別精度が期待できます"
    else:
        return "📊 標準: 一般的な識別精度です"

with st.sidebar:
    st.header("🎛️ 設定")
    
    # 区間長設定
    st.subheader("🕐 解析区間長")
    duration_type = st.selectbox(
        "区間長の設定",
        ["推奨5秒", "長め10秒", "短め2秒", "マニュアル"],
        index=0,
        help="推奨5秒: バランスの良い精度\n長め10秒: 最高精度\n短め2秒: 高速処理\nマニュアル: 自由設定"
    )
    
    # マニュアル入力の場合
    manual_duration = None
    if duration_type == "マニュアル":
        manual_duration = st.number_input(
            "区間長 [秒]",
            min_value=0.5,
            max_value=60.0,
            value=5.0,
            step=0.1,
            format="%.1f",
            help="0.5秒から60秒まで設定可能"
        )
    
    # 実際の区間長を計算
    window_sec = get_window_duration(duration_type, manual_duration)
    
    # 精度警告を表示
    warning_msg = get_accuracy_warning(window_sec)
    if "⚠️" in warning_msg:
        st.error(warning_msg)
    elif "⚡" in warning_msg:
        st.warning(warning_msg)
    elif "📊" in warning_msg:
        st.info(warning_msg)
    else:
        st.success(warning_msg)
    
    st.caption(f"設定された区間長: {window_sec:.1f}秒")
    
    # その他の設定
    st.subheader("🔧 詳細設定")
    hop_sec = st.number_input("内部フレームのホップ長 [秒]（参考）", value=DEFAULT_HOP_SEC, min_value=0.1, step=0.01, format="%.2f")
    agg_method = st.selectbox("スコア集約法", ["mean", "max"], index=0)
    topk = st.slider("表示クラス Top-K", 3, 10, 5)

st.write("**手順**：音声ファイル（WAV/MP3など）またはCSVファイルをアップロード → 波形確認 → 時刻選択 → 分類実行")

# ファイルタイプ選択
file_type = st.radio("ファイル形式を選択", ["CSVファイル", "音声ファイル"], index=0)

if file_type == "音声ファイル":
    uploaded = st.file_uploader("音声ファイル（WAV/MP3/M4A など）", type=["wav", "mp3", "m4a", "ogg", "flac"])
    
    if uploaded is not None:
        # 音声読み込み（librosaで一括）
        data, sr = librosa.load(io.BytesIO(uploaded.read()), sr=None, mono=False)
        y = ensure_mono_16k(data, sr)
        duration_total = len(y) / TARGET_SR
        timestamps = None
        start_time = None
        end_time = None
        
        st.caption(f"読み込み完了：{duration_total:.2f} 秒（16 kHz モノ化済）")
        
        # 区間長とファイル長の確認
        if duration_total < window_sec:
            st.error(f"❌ エラー: 設定された区間長（{window_sec:.1f}秒）がファイルの長さ（{duration_total:.2f}秒）より長くなっています。")
            st.error("区間長を短く設定してください。")
            st.stop()
        
        # 区間選択スライダー（秒数）
        start_sec = st.slider(
            "分析開始位置 [秒]",
            min_value=0.0,
            max_value=max(0.0, duration_total - window_sec),
            value=0.0,
            step=0.01
        )
        start_datetime = None
        
        # 波形表示
        fig = make_wave_plot(y, TARGET_SR, start_sec, window_sec)
        st.plotly_chart(fig, use_container_width=True)

else:  # CSVファイル
    uploaded = st.file_uploader("CSVファイル（waveform_*.csv）", type=["csv"])
    
    if uploaded is not None:
        # 最初にCSV全体を読み込んで日付範囲を取得
        y_all, estimated_sr, start_time_all, end_time_all, timestamps_all = load_csv_audio(uploaded)
        
        st.caption(f"ファイル全期間：{start_time_all.strftime('%Y-%m-%d %H:%M:%S')} から {end_time_all.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 年月日選択
        target_date = st.date_input(
            "表示する日付を選択",
            value=start_time_all.date(),
            min_value=start_time_all.date(),
            max_value=end_time_all.date()
        )
        
        # 指定日の datetime オブジェクトを作成
        target_datetime = datetime.combine(target_date, datetime.min.time())
        
        # 指定日のデータを読み込み
        if target_datetime.date() == start_time_all.date():
            # 同じ日付なら全体のデータをそのまま使用（重複処理を避ける）
            y, estimated_sr, start_time, end_time, timestamps = y_all, estimated_sr, start_time_all, end_time_all, timestamps_all
        else:
            y, estimated_sr, start_time, end_time, timestamps = load_csv_audio(uploaded, target_datetime)
        
        if len(y) == 0:
            st.warning(f"{target_date.strftime('%Y-%m-%d')} にはデータがありません。")
            start_datetime = None
            start_sec = 0.0
        else:
            y = ensure_mono_16k(y, estimated_sr)
            # 実際のサンプリングレートで継続時間を計算
            duration_total = len(y) / TARGET_SR
            
            st.caption(f"選択日のデータ：{start_time.strftime('%Y-%m-%d %H:%M:%S')} から {end_time.strftime('%Y-%m-%d %H:%M:%S')} までの {duration_total:.2f} 秒")
            
            # 区間長とファイル長の確認
            if duration_total < window_sec:
                st.error(f"❌ エラー: 設定された区間長（{window_sec:.1f}秒）がファイルの長さ（{duration_total:.2f}秒）より長くなっています。")
                st.error("区間長を短く設定してください。")
                st.stop()
            
            # 時刻入力（時分秒で指定）
            st.subheader("🕐 推論開始時刻を指定")
            
            if duration_total > window_sec:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    hour = st.number_input(
                        "時 (Hour)", 
                        min_value=start_time.hour, 
                        max_value=end_time.hour,
                        value=start_time.hour, 
                        step=1
                    )
                
                with col2:
                    # 時が変わった場合の分の範囲調整
                    if hour == start_time.hour:
                        min_minute = start_time.minute
                    else:
                        min_minute = 0
                    
                    if hour == end_time.hour:
                        max_minute = end_time.minute
                    else:
                        max_minute = 59
                    
                    minute = st.number_input(
                        "分 (Minute)", 
                        min_value=min_minute, 
                        max_value=max_minute,
                        value=min_minute, 
                        step=1
                    )
                
                with col3:
                    # 時分が変わった場合の秒の範囲調整
                    if hour == start_time.hour and minute == start_time.minute:
                        min_second = start_time.second
                    else:
                        min_second = 0
                    
                    if hour == end_time.hour and minute == end_time.minute:
                        max_second = end_time.second
                    else:
                        max_second = 59
                    
                    second = st.number_input(
                        "秒 (Second)", 
                        min_value=min_second, 
                        max_value=max_second,
                        value=min_second, 
                        step=1
                    )
                
                # 指定された時刻でdatetimeオブジェクトを作成
                try:
                    start_datetime = start_time.replace(hour=hour, minute=minute, second=second, microsecond=0)
                    
                    # 指定時刻が範囲内かチェック
                    if start_datetime < start_time:
                        start_datetime = start_time
                        st.warning("指定時刻が範囲外です。開始時刻に設定されました。")
                    elif start_datetime > end_time - timedelta(seconds=window_sec):
                        start_datetime = end_time - timedelta(seconds=window_sec)
                        st.warning("指定時刻が範囲外です。有効な最大時刻に設定されました。")
                        
                except ValueError as e:
                    start_datetime = start_time
                    st.error(f"無効な時刻です: {e}")
                
                st.success(f"🎯 推論開始時刻: {start_datetime.strftime('%H:%M:%S')}")
                
            else:
                start_datetime = start_time
                st.info(f"データ長が推論区間（{window_sec:.2f}秒）より短いため、全区間を使用します。")
            
            start_sec = 0.0  # CSVモードでは使用しない
            
            # 波形表示（日時軸）
            fig = make_wave_plot(y, TARGET_SR, 0, window_sec, timestamps, start_datetime)
            st.plotly_chart(fig, use_container_width=True)

# 音声またはCSVファイルがアップロードされた場合の処理
if uploaded is not None:
    # モデル読み込み
    with st.spinner("YAMNetをロード中..."):
        model, class_names = load_yamnet()

    # 推論ボタン
    if st.button("この区間を分類", type="primary"):
        # CSVファイルの場合は日時から区間を抽出
        if file_type == "CSVファイル" and start_datetime is not None:
            segment = slice_by_datetime(y, timestamps, start_datetime, window_sec)
        else:
            segment = slice_by_time(y, start_sec, window_sec, sr=TARGET_SR)
        
        # 入力は [samples] -> [N] の shape; TFは [N] でOK
        waveform = tf.convert_to_tensor(segment, dtype=tf.float32)

        with st.spinner("推論中..."):
            scores, embeddings, spectrogram = run_yamnet(model, waveform)

        # 時系列ヒートマップ（上位クラスだけ）
        agg_scores = aggregate_scores(scores, method=agg_method)
        top_idx = np.argsort(agg_scores)[::-1][:topk]
        top_labels = [class_names[i] for i in top_idx]
        top_labels_jp = [get_japanese_class_name(label) for label in top_labels]
        top_probs = agg_scores[top_idx]

        st.subheader("推定結果（選択区間の集約）")
        
        # 分析時刻の表示
        if file_type == "CSVファイル" and start_datetime is not None:
            st.caption(f"分析時刻: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')} から {window_sec:.2f}秒間")
        else:
            st.caption(f"分析区間: {start_sec:.2f}秒 から {start_sec + window_sec:.2f}秒")
            
        df = pd.DataFrame({
            "順位": np.arange(1, len(top_idx)+1),
            "音響クラス（日本語）": top_labels_jp,
            "音響クラス（英語）": top_labels,
            "スコア": top_probs
        })
        st.dataframe(df, use_container_width=True)

        # 参考：フレーム時系列の可視化（上位クラスのみ）
        frame_times = yamnet_frame_times(scores.shape[0], hop_sec=hop_sec)
        st.subheader("時系列スコア（Top-K クラス）")
        st.caption("YAMNetは約0.96秒のフレームで0.48秒ずつずらして出力します。")
        heat = make_scores_heatmap(frame_times, scores, top_idx, class_names)
        st.plotly_chart(heat, use_container_width=True)

        # 音声の再生（区間）- 音声ファイルの場合のみ
        if file_type == "音声ファイル":
            st.subheader("選択区間のプレビュー")
            # WAVにして一時バッファへ
            buf = io.BytesIO()
            sf.write(buf, segment, TARGET_SR, format="WAV")
            st.audio(buf.getvalue(), format="audio/wav")

else:
    st.info("ファイルをアップロードしてください。")
