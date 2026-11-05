# Nguyen Phung Nhat Khanh · Academic portfolio

A statically exported Next.js portfolio for GitHub Pages. Articles, study notes,
and project descriptions are stored in Markdown. No database or CMS is required.

## Local development

Use Node.js 22 (the same version as CI).

```sh
npm ci
npm run dev
```

Run `npm run build` to type-check and generate the static website in `out/`.
GitHub Actions builds and deploys it when changes are pushed to `main`.
To preview the export, serve `out/` with a static HTTP server.

## Editing the portfolio

- Profile, GitHub URL, and interests: `src/lib/site.ts`; introduction: `src/app/page.tsx`.
- Design tokens and shared layouts: `src/app/globals.css`.
- Writing: `content/blog/<slug>.md`.
- Projects: `content/projects/<slug>/index.md`.
- Books: `content/notes/<book>/_book.json`.
- Notes: `content/notes/<book>/<chapter>/<slug>.md`.
- Images: `public/images/`, referenced as `/images/filename.jpg`.

Blog posts and notes require a title and a quoted ISO date (`YYYY-MM-DD`).
Descriptions appear in listings and search previews. Notes use a numeric
`chapter` field to determine their position in the book; their folder remains
part of the URL. Available notes are discovered from files, so there is no
separate availability flag to maintain.

The Training Linear Models note lives at
`chapter-04/traning_linear_model`, matching its chapter 4 metadata.

The shared Markdown renderer normalizes headings below the page title,
generates section anchors and a table of contents, highlights code, and renders
math. It rejects missing local images. Add image dimensions in
`src/lib/markdown.ts` when adding new illustrations to reserve layout space.
Use dollar delimiters for inline and display math.

Project repository/demo links appear only when they contain a usable HTTP(S)
URL. Add real repository and demo URLs when publishing a project.
No university email, CV, evaluation results, or other credentials are invented.

## Deployment

This is a user site at `https://07-khanh.github.io`; no `basePath` is needed.
Static exports use directory-style URLs for direct links on GitHub Pages.
Canonical URLs and the sitemap use the origin in `src/lib/site.ts`.

The existing Next.js 14 / React 18 stack is retained. The dependency audit reports
upstream advisories; a coordinated framework and lint-tooling upgrade remains a
separate maintenance task. GitHub Pages serves only the generated static files.
