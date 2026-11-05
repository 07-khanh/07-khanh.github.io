import type { MetadataRoute } from 'next';
import { getBookNotes, getBooks, getPosts, notePath } from '@/lib/content';
import { site } from '@/lib/site';

export const dynamic = 'force-static';

export default function sitemap(): MetadataRoute.Sitemap {
  return [
    ...['/', '/projects/', '/blog/', '/notes/'].map(route => ({ url: `${site.url}${route}` })),
    ...getPosts().map(post => ({ url: `${site.url}/blog/${post.slug}/` })),
    ...getBooks().flatMap(book => [
      { url: `${site.url}/notes/${book.slug}/` },
      ...getBookNotes(book.slug).map(note => ({ url: `${site.url}${notePath(book.slug, note)}/` })),
    ]),
  ];
}
