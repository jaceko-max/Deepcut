const repoUrl = process.env.NEXT_PUBLIC_REPO_URL;

export function Footer() {
  return (
    <footer className="mt-24 border-t border-line">
      <div className="mx-auto flex max-w-3xl flex-col gap-2 px-4 py-8 text-sm text-muted sm:flex-row sm:items-center sm:justify-between">
        <p>Previews via Spotify &amp; Deezer · Data from Last.fm</p>
        {repoUrl && (
          <a
            className="inline-flex min-h-11 items-center underline decoration-line underline-offset-4 hover:text-fg"
            href={repoUrl}
            target="_blank"
            rel="noreferrer"
          >
            Source on GitHub
          </a>
        )}
      </div>
    </footer>
  );
}
