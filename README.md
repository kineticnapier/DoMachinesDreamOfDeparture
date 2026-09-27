# 機械たちは旅立ちの夢を見るか？

**Do Machines Dream of Departure?**

人間と同等の知覚・身体的制約を与えた学習機械に A Dance of Fire and Ice (ADOFAI) をプレイさせ、人間とは異なる有効な運指を自力で獲得できるかを調べる実験プロジェクトです。

This is an experimental project that trains an agent to play A Dance of Fire and Ice (ADOFAI) under human-like perceptual and physical constraints, with the goal of investigating whether it can discover effective fingering strategies different from those used by humans.

> 人間より速い機械を作るのではない。  
> 人間と同じ制約の中で、機械が何を学ぶのかを見る。
>
> The goal is not to build a machine that is simply faster than a human.  
> The goal is to see what a machine learns when it is given comparable constraints.

## 研究課題 / Research Question

**人間と同等の知覚・身体的制約を与えられた学習機械は、ADOFAI において人間とは異なる有効な運指を獲得できるか？**

**Can a learning agent, given human-like perceptual and physical constraints, acquire effective ADOFAI fingering strategies that differ from those discovered by humans?**

## 方針 / Principles

エージェントにはキー入力を直接生成させません。エージェントが出力できるのは指に対する運動指令だけであり、入力が成立するまでには筋肉応答、指の運動、キースイッチの作動という経路を通ります。

The agent does not directly emit key presses. Its actions are motor commands to simulated fingers; a key press occurs only after passing through muscle response, finger motion, and keyboard-switch actuation.

また、本番の実験では譜面ファイルから正解の入力時刻を直接取得させません。人間と同様、利用可能な知覚情報から未来を予測して行動することを目標とします。

In the final experiments, the agent will not receive exact target timings from the level file. Like a human player, it must predict future actions from the perceptual information available to it.

## 開発段階 / Roadmap

v0.1 では人体の完全な再現を目指しません。まず2本の指を持つ簡略化した身体モデルを作り、単指では物理的に突破できない速度に対して、エージェントが教示なしで交互押しなどの運指を獲得するかを検証します。

v0.1 does not attempt to reproduce the entire human body. It begins with a simplified two-finger model and tests whether the agent independently discovers alternating or other useful fingering patterns when a sequence is physically impossible to clear with a single finger.

詳細は [`docs/specification.md`](docs/specification.md) を参照してください。

See [`docs/specification.md`](docs/specification.md) for the current specification.

## 最終章 / Final Chapter

> **機械たちは夢の終わりに星を見る**

The final experiment is reserved for the point at which a human-constrained agent reaches the frontier of human ADOFAI play—or finds a physically reproducible route beyond it.

## 名称について / About the Title

プロジェクト名および最終章名は『ブルーアーカイブ』の募集名に着想を得ています。本プロジェクトは Yostar、NEXON Games、7th Beat Games その他の各権利者による公式プロジェクトではなく、提携・承認を受けたものでもありません。

The project title and final-chapter title are inspired by recruitment titles from *Blue Archive*. This is an independent project and is not affiliated with, endorsed by, or an official project of Yostar, NEXON Games, 7th Beat Games, or their respective rights holders.
