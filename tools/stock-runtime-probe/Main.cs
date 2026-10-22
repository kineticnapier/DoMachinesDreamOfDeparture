using HarmonyLib;
using Newtonsoft.Json;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Threading;
using UnityEngine;
using UnityModManagerNet;

namespace Dmdod.StockRuntimeProbe
{
    public static class Main
    {
        private const string ExpectedDllSha256 = "0D1524DABF0F8C67AA58BA1992D6ED858522B9094490EBBB00E5E8AEFF7F40CD";
        private static Harmony harmony;
        private static ProbeSink sink;

        public static bool Load(UnityModManager.ModEntry entry)
        {
            try
            {
                var config = ProbeConfig.Load(Path.Combine(entry.Path, "probe-config.json"));
                var assemblyPath = typeof(scrPlayer).Assembly.Location;
                var actualHash = Sha256(assemblyPath);
                var outputDir = Path.Combine(entry.Path, "runtime-traces");
                Directory.CreateDirectory(outputDir);
                var sessionId = DateTime.UtcNow.ToString("yyyyMMddTHHmmssfffZ") + "-" + Guid.NewGuid().ToString("N").Substring(0, 8);
                var outputPath = Path.Combine(outputDir, "stock-runtime-" + sessionId + ".jsonl");
                sink = new ProbeSink(outputPath, sessionId, config, actualHash, Application.version, entry.Info.Version.ToString());
                Probe.Sink = sink;
                harmony = new Harmony(entry.Info.Id);
                PatchInstaller.Install(harmony);
                entry.OnToggle = OnToggle;
                entry.OnUnload = OnUnload;
                entry.Logger.Log("read-only trace: " + outputPath);
                if (!string.Equals(actualHash, ExpectedDllSha256, StringComparison.OrdinalIgnoreCase))
                    entry.Logger.Warning("Assembly-CSharp hash differs from the audited DLL; trace remains usable but parity must be UNVERIFIED.");
                return true;
            }
            catch (Exception exception)
            {
                entry.Logger.Error("probe load failed: " + exception);
                SafeDispose();
                return false;
            }
        }

        private static bool OnToggle(UnityModManager.ModEntry entry, bool value)
        {
            if (sink != null)
                sink.Enabled = value;
            return true;
        }

        private static bool OnUnload(UnityModManager.ModEntry entry)
        {
            try { harmony?.UnpatchAll(entry.Info.Id); } catch { }
            SafeDispose();
            return true;
        }

        private static void SafeDispose()
        {
            try { sink?.Dispose(); } catch { }
            sink = null;
            Probe.Sink = null;
        }

        private static string Sha256(string path)
        {
            using (var stream = File.OpenRead(path))
            using (var hash = SHA256.Create())
                return BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "");
        }
    }

    internal sealed class ProbeConfig
    {
        public bool enabled = true;
        public int minimumFloorInclusive = 34;
        public int maximumFloorInclusive = 40;
        public int maximumQueuedLines = 8192;
        public int flushIntervalMs = 100;

        public static ProbeConfig Load(string path)
        {
            try
            {
                if (File.Exists(path))
                    return JsonConvert.DeserializeObject<ProbeConfig>(File.ReadAllText(path)) ?? new ProbeConfig();
            }
            catch { }
            return new ProbeConfig();
        }
    }

    internal static class PatchInstaller
    {
        private static readonly Type NullableUlong = typeof(ulong?);

        public static void Install(Harmony harmony)
        {
            Patch(harmony, typeof(scrPlayer), "Simulated_PlayerControl_Update", new[] { NullableUlong }, "SimulatedUpdatePrefix", "SimulatedUpdatePostfix");
            Patch(harmony, typeof(scrPlayer), "CheckPostHoldFail", new[] { NullableUlong }, "CheckPostHoldFailPrefix", "CheckPostHoldFailPostfix");
            Patch(harmony, typeof(scrController), "ProcessKeyInputs", new[] { typeof(ulong) }, "ProcessKeyInputsPrefix", null);
            Patch(harmony, typeof(scrPlayer), "HitInputEvent", new[] { typeof(bool), typeof(InputEventState) }, "HitInputEventPrefix", null);
            Patch(harmony, typeof(scrPlanet), "SwitchChosen", Type.EmptyTypes, "SwitchChosenPrefix", "SwitchChosenPostfix");
            Patch(harmony, typeof(scrMarginTracker), "AddHit", new[] { typeof(HitMargin) }, "AddHitPrefix", "AddHitPostfix");
            Patch(harmony, typeof(scrHitTextManager), "ShowHitText", new[] { typeof(HitMargin), typeof(scrPlanet), typeof(float) }, "ShowHitTextPrefix", null);
            Patch(harmony, typeof(scrPlayer), "Die", new[] { typeof(bool), typeof(bool), typeof(string), typeof(bool) }, "DiePrefix", "DiePostfix");
            Patch(harmony, typeof(scrController), "OnPlayerDied", new[] { typeof(scrPlayer), typeof(bool), typeof(bool), typeof(string), typeof(bool) }, null, "OnPlayerDiedPostfix");
        }

        private static void Patch(Harmony harmony, Type type, string name, Type[] arguments, string prefixName, string postfixName)
        {
            var original = AccessTools.Method(type, name, arguments);
            if (original == null)
                throw new MissingMethodException(type.FullName, name);
            var flags = BindingFlags.Static | BindingFlags.NonPublic;
            var prefix = prefixName == null ? null : new HarmonyMethod(typeof(Patches).GetMethod(prefixName, flags));
            var postfix = postfixName == null ? null : new HarmonyMethod(typeof(Patches).GetMethod(postfixName, flags));
            harmony.Patch(original, prefix, postfix);
        }
    }

    internal static class Patches
    {
        private static void SimulatedUpdatePrefix(scrPlayer __instance, out long __state)
        {
            __state = Probe.Begin("frame_begin", "Simulated_PlayerControl_Update", __instance);
        }

        private static void SimulatedUpdatePostfix(scrPlayer __instance, long __state)
        {
            Probe.End("frame_end", "Simulated_PlayerControl_Update", __instance, __state);
        }

        private static void CheckPostHoldFailPrefix(scrPlayer __instance, out long __state)
        {
            __state = Probe.Begin("check_post_hold_fail", "CheckPostHoldFail", __instance, "prefix");
        }

        private static void CheckPostHoldFailPostfix(scrPlayer __instance, long __state)
        {
            Probe.End("check_post_hold_fail", "CheckPostHoldFail", __instance, __state);
        }

        private static void ProcessKeyInputsPrefix(object __instance, ulong eventTick)
        {
            Probe.Emit("input_dispatch", "ProcessKeyInputs", __instance, new Dictionary<string, object> { ["event_tick"] = eventTick });
        }

        private static void HitInputEventPrefix(scrPlayer __instance, bool isAuto, InputEventState state)
        {
            if (!isAuto && state == InputEventState.Down)
                Probe.Emit("key_down", "HitInputEvent", __instance, new Dictionary<string, object> { ["input_state"] = state.ToString(), ["is_auto"] = false });
        }

        private static void SwitchChosenPrefix(scrPlanet __instance, out int __state)
        {
            __state = Probe.ReadFloor(__instance);
            Probe.Emit("judgement", "SwitchChosen", __instance, new Dictionary<string, object> { ["hook_phase"] = "prefix", ["floor_before_evaluator_only"] = __state });
        }

        private static void SwitchChosenPostfix(scrPlanet __instance, scrPlanet __result, int __state)
        {
            Probe.Emit("judgement", "SwitchChosen", __instance, new Dictionary<string, object> { ["hook_phase"] = "postfix", ["floor_before_evaluator_only"] = __state, ["floor_after_evaluator_only"] = Probe.ReadFloor(__result) });
        }

        private static void AddHitPrefix(scrMarginTracker __instance, HitMargin hit)
        {
            Probe.Emit(hit == HitMargin.FailMiss ? "auto_fail_miss" : "judgement", "AddHit", __instance, new Dictionary<string, object> { ["hook_phase"] = "prefix", ["judgement"] = hit.ToString() });
        }

        private static void AddHitPostfix(scrMarginTracker __instance, HitMargin hit)
        {
            var counts = new Dictionary<string, int>();
            foreach (HitMargin value in Enum.GetValues(typeof(HitMargin)))
            {
                try { counts[value.ToString()] = __instance.GetHits(value); } catch { }
            }
            Probe.Emit("add_hit", "AddHit", __instance, new Dictionary<string, object> {
                ["hook_phase"] = "postfix",
                ["judgement"] = hit.ToString(),
                ["counter_snapshot_internal_evaluator_only"] = counts,
                ["visible_counter_state"] = null,
            });
        }

        private static void ShowHitTextPrefix(object __instance, HitMargin hitMargin, scrPlanet planet, float missAngle)
        {
            Probe.Emit("visible_judgement_signal", "ShowHitText", planet, new Dictionary<string, object> { ["judgement"] = hitMargin.ToString(), ["miss_angle"] = missAngle });
        }

        private static void DiePrefix(scrPlayer __instance, bool overload, bool multipress, string failMessage, bool hitbox, out long __state)
        {
            __state = Probe.Begin("game_stop_requested", "Die", __instance, "prefix", new Dictionary<string, object> { ["overload"] = overload, ["multipress"] = multipress, ["hitbox"] = hitbox });
        }

        private static void DiePostfix(scrPlayer __instance, long __state)
        {
            Probe.End("game_stop_requested", "Die", __instance, __state);
        }

        private static void OnPlayerDiedPostfix(scrPlayer deadPlayer, bool overload, bool multipress, string failMessage, bool hitbox)
        {
            Probe.Emit("game_stop", "OnPlayerDied", deadPlayer, new Dictionary<string, object> { ["overload"] = overload, ["multipress"] = multipress, ["hitbox"] = hitbox });
        }
    }

    internal static class Probe
    {
        public static ProbeSink Sink;

        public static long Begin(string kind, string hook, object instance, string phase = "prefix", IDictionary<string, object> payload = null)
        {
            var started = Stopwatch.GetTimestamp();
            Emit(kind, hook, instance, Merge(payload, "hook_phase", phase));
            return started;
        }

        public static void End(string kind, string hook, object instance, long started)
        {
            var microseconds = started == 0 ? (double?)null : (Stopwatch.GetTimestamp() - started) * 1000000.0 / Stopwatch.Frequency;
            Emit(kind, hook, instance, new Dictionary<string, object> { ["hook_phase"] = "postfix", ["hook_duration_us"] = microseconds });
        }

        public static void Emit(string kind, string hook, object instance, IDictionary<string, object> payload = null)
        {
            try
            {
                var target = Sink;
                if (target == null || !target.Enabled)
                    return;
                var floor = ReadFloor(instance);
                if (!target.InScope(floor))
                    return;
                var row = payload == null ? new Dictionary<string, object>() : new Dictionary<string, object>(payload);
                row["hook"] = hook;
                row["floor_evaluator_only"] = floor >= 0 ? (object)floor : null;
                row["no_fail"] = ReadNoFail();
                target.Emit(kind, row);
            }
            catch (Exception exception)
            {
                try { Sink?.EmitHookError(hook, exception); } catch { }
            }
        }

        public static int ReadFloor(object instance)
        {
            try
            {
                object value = instance;
                foreach (var name in new[] { "player", "currFloor", "currfloor" })
                {
                    if (value == null) break;
                    var type = value.GetType();
                    var member = (MemberInfo)AccessTools.Field(type, name) ?? AccessTools.Property(type, name);
                    if (member == null) continue;
                    value = member is FieldInfo field ? field.GetValue(value) : ((PropertyInfo)member).GetValue(value, null);
                    if (value != null && value.GetType().Name == "scrFloor")
                    {
                        var seq = AccessTools.Field(value.GetType(), "seqID")?.GetValue(value);
                        if (seq != null) return Convert.ToInt32(seq);
                    }
                }
                var player = ADOBase.controller?.playerOne;
                return player?.currFloor == null ? -1 : player.currFloor.seqID;
            }
            catch { return -1; }
        }

        private static bool ReadNoFail()
        {
            try { return ADOBase.controller != null && ADOBase.controller.noFail; }
            catch { return false; }
        }

        private static IDictionary<string, object> Merge(IDictionary<string, object> source, string key, object value)
        {
            var result = source == null ? new Dictionary<string, object>() : new Dictionary<string, object>(source);
            result[key] = value;
            return result;
        }
    }

    internal sealed class ProbeSink : IDisposable
    {
        private readonly string path;
        private readonly string sessionId;
        private readonly ProbeConfig config;
        private readonly ConcurrentQueue<string> queue = new ConcurrentQueue<string>();
        private readonly AutoResetEvent wake = new AutoResetEvent(false);
        private readonly Thread worker;
        private long sequence;
        private long dropped;
        private volatile bool stopping;
        public bool Enabled { get; set; }

        public ProbeSink(string path, string sessionId, ProbeConfig config, string dllHash, string gameBuild, string modVersion)
        {
            this.path = path;
            this.sessionId = sessionId;
            this.config = config;
            Enabled = config.enabled;
            queue.Enqueue(JsonConvert.SerializeObject(new Dictionary<string, object> {
                ["type"] = "stock_runtime_probe_header", ["session_id"] = sessionId,
                ["evidence_kind"] = "live_runtime", ["assembly_csharp_sha256"] = dllHash,
                ["game_build"] = gameBuild, ["probe_version"] = modVersion,
                ["created_utc"] = DateTime.UtcNow.ToString("O"), ["student_visible_fields"] = new object[0],
            }));
            worker = new Thread(WriteLoop) { IsBackground = true, Name = "DMDOD Stock Runtime Probe Writer" };
            worker.Start();
        }

        public bool InScope(int floor) => floor < 0 || (floor >= config.minimumFloorInclusive && floor <= config.maximumFloorInclusive);

        public void Emit(string kind, IDictionary<string, object> payload)
        {
            if (queue.Count >= config.maximumQueuedLines)
            {
                Interlocked.Increment(ref dropped);
                return;
            }
            var row = new Dictionary<string, object>(payload) {
                ["type"] = "event", ["session_id"] = sessionId,
                ["sequence"] = Interlocked.Increment(ref sequence),
                ["frame"] = Time.frameCount, ["realtime_s"] = (double)Time.realtimeSinceStartup,
                ["kind"] = kind, ["dropped_before"] = Interlocked.Read(ref dropped),
            };
            queue.Enqueue(JsonConvert.SerializeObject(row));
            wake.Set();
        }

        public void EmitHookError(string hook, Exception exception)
        {
            Emit("hook_error", new Dictionary<string, object> { ["hook"] = hook, ["error_type"] = exception.GetType().FullName, ["error"] = exception.Message, ["no_fail"] = false });
        }

        private void WriteLoop()
        {
            try
            {
                using (var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.Read))
                using (var writer = new StreamWriter(stream))
                {
                    while (!stopping || !queue.IsEmpty)
                    {
                        while (queue.TryDequeue(out var line)) writer.WriteLine(line);
                        writer.Flush();
                        if (!stopping) wake.WaitOne(Math.Max(10, config.flushIntervalMs));
                    }
                }
            }
            catch { }
        }

        public void Dispose()
        {
            stopping = true;
            wake.Set();
            if (worker != null && worker.IsAlive) worker.Join(2000);
            wake.Dispose();
        }
    }
}
