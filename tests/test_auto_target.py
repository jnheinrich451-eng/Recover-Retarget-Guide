"""Known answers for the experiment set's fixed caption rule (pure text; no model)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.anchor.m2d.auto_target import arbiter_label, scene_from_caption, target_for  # noqa: E402


def main():
    s = scene_from_caption("A warrior in armor walks through a serene village by a river, surrounded by lush greenery and rustic buildings.")
    assert s["scene"] == "through a serene village by a river", s                       # cut at ', surrounded'
    s = scene_from_caption("A character rides a horse through a cobblestone street lined with old buildings and debris under a cloudy sky.")
    assert s["scene"].startswith("through a cobblestone street"), s                     # the verb's object is skipped to the locative
    s = scene_from_caption("A man in a blue robe walks through a grand, ornate greenhouse filled with lush plants and intricate architecture.")
    assert s["scene"].startswith("through a grand, ornate greenhouse"), s               # an adjective comma does not cut
    s = scene_from_caption("A warrior and his son navigate a misty forest path, preparing for an unseen challenge ahead.")
    assert s["scene"] == "a misty forest path", s                                       # no locative after the verb: the plain remainder, cut at ', preparing'
    s = scene_from_caption("Several goldfish swim among green plants in a blue aquarium")
    assert s["scene"] == "among green plants in a blue aquarium", s
    s = scene_from_caption("A quiet harbour at dawn")
    assert s["scene"] == "at dawn" and s["scene_source"].startswith("locative"), s     # no listed verb: from the first locative
    s = scene_from_caption("Nothing moves here")
    assert s["scene_source"].startswith("verb 'moves'") and s["scene"] == "here", s
    t = target_for("vehicle", "A sleek blue sports car drives through a sunny cityscape with palm trees and modern architecture.")
    assert (t["object"], t["action"]) == ("a red car", "driving") and t["scene"].startswith("through a sunny cityscape"), t
    assert arbiter_label("vehicle", "A horse-drawn carriage travels through a cobblestone street") == "horse"
    assert arbiter_label("human", "A camel walks slowly across a sandy enclosure") == "animal"
    assert arbiter_label("human", "A rider on a BMX bike launches over dirt jumps on a track") == "bicycle"   # the machine before its rider
    assert arbiter_label("human", "A character runs through a bustling, historical cityscape with horse-drawn carriages") == "person"   # background nouns after the verb do not label the subject
    assert arbiter_label("human", "A lone rider on horseback traverses a lush green field") == "horse"
    assert arbiter_label("human", "A man in a suit walks down a cobblestone street") == "person"
    print("test_auto_target: ok")


if __name__ == "__main__":
    main()
