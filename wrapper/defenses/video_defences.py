from utils.io_utils import read_video_frames, write_video
from defenses.adaptive_defense import AdaptiveDefense
from defenses.attack_classifier import AttackClassifier


def defend_attacked_video(
    attacked_video_path: str,
    output_video_path: str = None,
    frame_skip: int = 1
) -> dict:
    """
    Детектит тип атаки на уже атакованном видео и применяет защиту.
    """
    frames, info = read_video_frames(attacked_video_path, rgb=True)
    
    defender = AdaptiveDefense()
    
    # Динамический словарь для любых типов атак
    stats = {
        "total_frames": len(frames),
        "processed_frames": 0,
        "detections": {}  # пустой словарь, будет заполняться динамически
    }
    
    defended_frames = []
    prev_frame_for_classification = None
    
    print(f"Processing video: {len(frames)} frames")
    
    for idx, frame in enumerate(frames):
        if idx % frame_skip == 0:
            attack_type = AttackClassifier.classify(
                frame,
                prev_frame=prev_frame_for_classification,
            )
            
            # Динамическое добавление новых типов атак
            if attack_type not in stats["detections"]:
                stats["detections"][attack_type] = 0
            
            stats["detections"][attack_type] += 1
            stats["processed_frames"] += 1
            defended_frame = defender.apply_with_type(frame, attack_type)
            
            if stats["processed_frames"] % 10 == 0:
                print(f"Frame {stats['processed_frames']}/{stats['total_frames']} - Attack: {attack_type}")
        else:
            defended_frame = frame
        
        defended_frames.append(defended_frame)
        prev_frame_for_classification = frame
    
    if output_video_path:
        write_video(
            path=output_video_path,
            frames=defended_frames,
            fps=info["fps"],
            width=info["width"],
            height=info["height"],
            rgb=True,
        )
        print(f"\nSaved defended video to: {output_video_path}")
    
    print("\n=== STATISTICS ===")
    for attack, count in stats["detections"].items():
        percent = count / stats["processed_frames"] * 100
        print(f"{attack}: {count} frames ({percent:.1f}%)")
    
    return stats


if __name__ == "__main__":
    # Пример запуска
    result = defend_attacked_video(
        attacked_video_path="../data/attack_contrast_attack_20260512_000725.mp4",  
        output_video_path="../results/defended_video.mp4",
        frame_skip=2
    )
