import type { Metadata } from 'next';
import { site } from './site';

export function pageMetadata(title: string, description: string, path: string, date?: string): Metadata {
  const fullTitle = `${title} · ${site.name}`;
  const canonical = path.endsWith('/') ? path : `${path}/`;
  return {
    title, description, alternates: { canonical },
    openGraph: {
      title: fullTitle, description, url: canonical, siteName: site.name,
      ...(date ? { type: 'article' as const, publishedTime: date, authors: [site.name] } : { type: 'website' as const }),
    },
    twitter: { card: 'summary', title: fullTitle, description },
  };
}
