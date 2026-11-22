import { formatBpm } from "@/lib/format";
import type { PlaylistInfo, PlaylistProfile as Profile } from "@/lib/types";
import { VibeCard } from "./VibeCard";

export function PlaylistProfile({ playlist, profile }: { playlist: PlaylistInfo | null; profile: Profile }) {
  return (
    <section aria-labelledby="profile-h" className="space-y-4">
      <div className="flex items-center gap-4">
        {playlist?.image_url && (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={playlist.image_url} alt="" width={64} height={64} className="size-16 rounded-md bg-surface-2 object-cover" />
        )}
        <div className="min-w-0">
          <h2 id="profile-h" className="display truncate text-2xl font-semibold">
            {playlist?.name ?? "Your playlist"}
          </h2>
          <p className="num text-sm text-muted">
            {playlist?.owner ? `by ${playlist.owner} · ` : ""}
            {playlist ? `${playlist.analyzed_tracks} of ${playlist.total_tracks} tracks analyzed · ` : ""}
            median {formatBpm(profile.bpm_median)} BPM ({formatBpm(profile.bpm_range[0])}–{formatBpm(profile.bpm_range[1])})
          </p>
        </div>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        {profile.vibes.map((v) => (
          <VibeCard key={v.id} vibe={v} />
        ))}
      </div>
    </section>
  );
}
