export const site = {
  name: 'Nguyen Phung Nhat Khanh',
  url: 'https://07-khanh.github.io',
  github: 'https://github.com/07-khanh',
  description: 'Nguyen Phung Nhat Khanh — HCMUS undergraduate interested in AI, machine learning, deep learning, and computer vision. Projects, technical writing, and study notes.',
  interests: ['Artificial Intelligence', 'Machine Learning', 'Deep Learning', 'Computer Vision'],
};

export function formatDate(date: string) {
  return new Intl.DateTimeFormat('en', {
    day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC',
  }).format(new Date(`${date}T00:00:00Z`));
}

export function publicProjectLink(url: string) {
  return /^https?:\/\//.test(url) && !url.startsWith('https://github.com/example/');
}
